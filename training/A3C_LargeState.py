import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

from environment.dynamic_env import DynamicTaxiEnv
from self_defined_state import StateRecorder_LargeState  # Use the redesigned StateRecorder_LargeState below

# --- Hyperparameters ---
NUM_EPISODES = 50000
NUM_WORKERS = 8
T_MAX = 5000
GAMMA = 0.99
LR = 1e-2
MAX_FUEL = 5000
BETA = 0.01  # Entropy coefficient

# --- Global network definition (Actor–Critic) ---
# Note: The input dimension is updated to 18.
class A3CNet(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=128, num_actions=6):
        super(A3CNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_policy = nn.Linear(hidden_dim, num_actions)
        self.fc_value = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        policy_logits = self.fc_policy(x)
        value = self.fc_value(x)
        return policy_logits, value

def ensure_shared_grads(local_net, global_net):
    """Copy local gradients to global network."""
    for local_param, global_param in zip(local_net.parameters(), global_net.parameters()):
        if global_param.grad is None:
            global_param.grad = local_param.grad

def potential(state):
    """
    Compute a potential for shaping rewards.
    If the passenger is not onboard (passenger_on flag == 0) and passenger is known,
    use the negative Manhattan distance (in absolute units) to the passenger.
    If the passenger is onboard (passenger_on flag == 1) and destination is known,
    use the negative Manhattan distance to the destination.
    Otherwise, potential is 0.
    Note: The relative differences were normalized by 10, so we multiply back by 10.
    """
    # Unpack state: indices based on our design.
    # state indices:
    # 2: passenger_known, 3: pass_diff_r, 4: pass_diff_c,
    # 5: destination_known, 6: dest_diff_r, 7: dest_diff_c,
    # 8: passenger_on.
    passenger_known = state[2]
    pass_diff_r = state[3]
    pass_diff_c = state[4]
    destination_known = state[5]
    dest_diff_r = state[6]
    dest_diff_c = state[7]
    passenger_on = state[8]
    
    if passenger_on < 0.5 and passenger_known == 1.0:
        # Use passenger distance.
        manhattan = 1 * (abs(pass_diff_r) + abs(pass_diff_c))
        return -manhattan
    elif passenger_on >= 0.5 and destination_known == 1.0:
        manhattan = 1 * (abs(dest_diff_r) + abs(dest_diff_c))
        return -manhattan
    else:
        return 0.0

def worker(worker_id, global_net, optimizer, global_ep, res_queue):
    # Create the environment and a StateRecorder_LargeState for this worker.
    env = DynamicTaxiEnv(grid_size_min=5, grid_size_max=10, fuel_limit=MAX_FUEL, obstacle_prob=0.10)
    local_net = A3CNet(input_dim=18, hidden_dim=128, num_actions=6)
    local_net.load_state_dict(global_net.state_dict())
    recorder = StateRecorder_LargeState(MAX_FUEL)
    
    while global_ep.value < NUM_EPISODES:
        obs, _ = env.reset()
        recorder.reset()  # Reset recorder at the start of the episode.
        state = recorder.get_state(obs)  # 18-dimensional state.
        done = False
        ep_reward = 0
        t = 0
        
        # Compute initial potential.
        phi_old = potential(state)
        
        # Lists for rollout.
        values, log_probs, rewards, entropies = [], [], [], []
        
        while not done and t < T_MAX:
            t += 1
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            policy_logits, value = local_net(state_tensor)
            probs = F.softmax(policy_logits, dim=1)
            log_prob = F.log_softmax(policy_logits, dim=1)
            
            # Entropy bonus for exploration.
            entropy = - (probs * log_prob).sum(dim=1).mean()
            entropies.append(entropy)
            
            # Sample an action.
            action = probs.multinomial(num_samples=1).detach()
            log_prob_action = log_prob.gather(1, action)
            action = int(action.item())
            
            # Update recorder and take a step.
            # Record passenger status before taking the action.
            was_passenger_on = recorder.passenger_on_taxi
            
            # Update recorder and take a step.
            recorder.update(obs, action)
            next_obs, reward, done, _ = env.step(action)
            
            # Check if the pickup action resulted in successfully picking up the passenger.
            now_passenger_on = recorder.passenger_on_taxi
            bonus = 0.0
            if action == 4 and (not was_passenger_on) and now_passenger_on:
                bonus = 10  # Add bonus shaping reward for successful pickup.

            next_state = recorder.get_state(next_obs)
            phi_new = potential(next_state)
            # Compute potential-based shaping.
            shaping = GAMMA * phi_new - phi_old
            phi_old = phi_new
            r_shaped = reward + shaping + bonus
            r_shaped = r_shaped / 50.0  # Normalize the reward.
            
            ep_reward += reward  # For reporting, we accumulate the raw reward.
            values.append(value)
            log_probs.append(log_prob_action)
            rewards.append(r_shaped)
            
            state = next_state
            obs = next_obs
        
        # Bootstrapped value.
        if done:
            R = 0
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            _, R = local_net(state_tensor)
            R = R.item()
        R = torch.tensor(R, dtype=torch.float32)
        
        policy_loss = 0
        value_loss = 0
        entropy_bonus = torch.stack(entropies).mean() if entropies else 0
        
        # Backpropagate through the rollout.
        for i in reversed(range(len(rewards))):
            R = rewards[i] + GAMMA * R
            advantage = R - values[i]
            value_loss += 0.5 * advantage.pow(2)
            policy_loss -= log_probs[i] * advantage.detach()
        total_loss = policy_loss + value_loss - BETA * entropy_bonus
        
        optimizer.zero_grad()
        total_loss.backward()
        ensure_shared_grads(local_net, global_net)
        optimizer.step()
        local_net.load_state_dict(global_net.state_dict())
        
        with global_ep.get_lock():
            global_ep.value += 1
        res_queue.put(ep_reward)
    
    res_queue.put(None)

def evaluate_agent(global_net, num_episodes=10):
    """
    Run full episodes using the current global network in evaluation mode.
    Uses greedy action selection (argmax over policy logits).
    Returns the average reward over the test episodes.
    """
    global_net.eval()
    test_env = DynamicTaxiEnv(grid_size_min=5, grid_size_max=10, fuel_limit=MAX_FUEL, obstacle_prob=0.10)
    test_rewards = []
    for ep in range(num_episodes):
        obs, _ = test_env.reset()
        recorder = StateRecorder_LargeState(MAX_FUEL)
        recorder.reset()
        total_reward = 0
        done = False
        while not done:
            state = recorder.get_state(obs)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                policy_logits, _ = global_net(state_tensor)
            action = int(torch.argmax(policy_logits, dim=1).item())
            recorder.update(obs, action)
            obs, reward, done, _ = test_env.step(action)
            total_reward += reward
        test_rewards.append(total_reward)
    return np.mean(test_rewards)

def a3c_train():
    global_net = A3CNet(input_dim=18, hidden_dim=128, num_actions=6)
    global_net.share_memory()
    optimizer = optim.Adam(global_net.parameters(), lr=LR)
    global_ep = mp.Value('i', 0)
    res_queue = mp.Queue()
    processes = []
    
    for worker_id in range(NUM_WORKERS):
        p = mp.Process(target=worker, args=(worker_id, global_net, optimizer, global_ep, res_queue))
        p.start()
        processes.append(p)
    
    rewards = []
    test_rewards = []
    while True:
        r = res_queue.get()
        if r is None:
            break
        rewards.append(r)
        if len(rewards) % 100 == 0:
            avg_train_reward = np.mean(rewards[-100:])
            avg_test_reward = evaluate_agent(global_net, num_episodes=10)
            test_rewards.append(avg_test_reward)
            print(f"After {len(rewards)} episodes, Avg Train Reward (last 100): {avg_train_reward:.2f} | Eval Avg Reward: {avg_test_reward:.2f}")
    
    for p in processes:
        p.join()
    
    return global_net, rewards, test_rewards

if __name__ == "__main__":
    mp.set_start_method('spawn')
    global_net, rewards, test_rewards = a3c_train()
    os.makedirs("./results_dynamic", exist_ok=True)
    torch.save(global_net.state_dict(), "./results_dynamic/a3c_policy_net.pt")
    np.save("./results_dynamic/a3c_rewards_history.npy", np.array(rewards))
    
    plt.figure(figsize=(10,6))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("A3C Reward History")
    plt.savefig("./results_dynamic/a3c_reward_history.png")
    plt.close()
    
    plt.figure(figsize=(10,6))
    episodes = range(0, len(test_rewards)*100, 100)
    plt.plot(episodes, test_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("A3C Test Reward History")
    plt.savefig("./results_dynamic/a3c_test_reward_history.png")
    plt.close()
