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
from self_defined_state import StateRecorder  # your class-based state recorder

# Hyperparameters
NUM_EPISODES = 50000
NUM_WORKERS = 8
T_MAX = 500
GAMMA = 0.99
LR = 1e-3
MAX_FUEL = 5000
BETA = 0.01  # entropy coefficient for exploration


# Global network definition (Actor–Critic)
class A3CNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128, num_actions=6):
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

def potential(state, env):
    """
    Compute a potential using the new state produced by StateRecorder.
    Here, we define potential as the negative of the sum of the absolute differences 
    in the "going" direction (i.e. the first two components of the new state), normalized by (env.grid_size-1).
    """
    # state is assumed to be:
    # (going_row_diff, going_col_diff, picked, last_action_norm, visited_code_norm, fuel_norm, obs_n, obs_s, obs_e, obs_w)
    return - (abs(state[0]) + abs(state[1])) / 20

def worker(worker_id, global_net, optimizer, global_ep, res_queue):
    # Each worker creates its own environment and its own StateRecorder.
    env = DynamicTaxiEnv(grid_size_min=5, grid_size_max=10, fuel_limit=MAX_FUEL, obstacle_prob=0.10)
    local_net = A3CNet(input_dim=10, hidden_dim=128, num_actions=6)
    local_net.load_state_dict(global_net.state_dict())
    recorder = StateRecorder(MAX_FUEL)
    
    while global_ep.value < NUM_EPISODES:
        obs, _ = env.reset()
        recorder.reset()  # Reset recorder at the start of the episode.
        state, other_state = recorder.get_state(obs)  # new state: 10-tuple
        done = False
        ep_reward = 0
        t = 0
        
        # Compute initial potential.
        phi_old = potential(state, env)
        
        # Lists for rollout.
        values, log_probs, rewards, entropies = [], [], [], []
        
        while not done and t < T_MAX:
            t += 1
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            policy_logits, value = local_net(state_tensor)
            probs = F.softmax(policy_logits, dim=1)
            log_prob = F.log_softmax(policy_logits, dim=1)

            # Compute entropy for exploration bonus.
            entropy = - (probs * log_prob).sum(dim=1).mean()
            entropies.append(entropy)

            # Sample an action.
            action = probs.multinomial(num_samples=1).detach()
            log_prob_action = log_prob.gather(1, action)
            action = int(action.item())
            
            # Update the recorder with current observation and chosen action.
            recorder.update(obs, action)
            next_obs, reward, done, _ = env.step(action)
            
            # Compute new state and potential.
            next_state, next_other_state = recorder.get_state(next_obs)
            phi_new = potential(next_state, env)
            # Compute shaping: potential-based shaping reward (scale factor 10, adjust as needed)
            shaping = (GAMMA * phi_new - phi_old) 

            # Override shaping if passenger has just been picked up.
            if state[2] == 0 and next_state[2] == 1 and action == 4:
                shaping = 10
                # print("Picked up passenger! Reward:", reward)
            elif action==4 and state[2]!=0:
                shaping = -10

            phi_old = phi_new
            r_shaped = reward + shaping
            # do normalaize the reward
            r_shaped = r_shaped / 50

            ep_reward += reward  # For reporting, you can also accumulate r_shaped.
            values.append(value)
            log_probs.append(log_prob_action)
            rewards.append(r_shaped)
            
            state = next_state
            other_state = next_other_state
            obs = next_obs
        
        # Bootstrapped value.
        if done:
            R = 0
        else:
            state_norm = recorder.normalize_state(state)
            state_tensor = torch.tensor(state_norm, dtype=torch.float32).unsqueeze(0)
            _, R = local_net(state_tensor)
            R = R.item()
        R = torch.tensor(R, dtype=torch.float32)
        
        policy_loss = 0
        value_loss = 0
        # Sum the entropy bonus over the rollout and take the mean.
        entropy_bonus = torch.stack(entropies).mean() if entropies else 0
        
        for i in reversed(range(len(rewards))):
            R = rewards[i] + GAMMA * R
            advantage = R - values[i]
            value_loss += 0.5 * advantage.pow(2)
            policy_loss -= log_probs[i] * advantage.detach()
        # Subtract the entropy bonus (to encourage exploration)
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
    Run full episodes using the current global network for evaluation.
    Uses greedy action selection (i.e., argmax of policy logits).
    Returns the average reward over the test episodes.
    """
    global_net.eval()
    test_env = DynamicTaxiEnv(grid_size_min=5, grid_size_max=10, fuel_limit=MAX_FUEL, obstacle_prob=0.10)
    test_rewards = []
    for ep in range(num_episodes):
        obs, _ = test_env.reset()
        recorder = StateRecorder(MAX_FUEL)
        recorder.reset()
        total_reward = 0
        done = False
        while not done:
            state, other_state = recorder.get_state(obs)
            state_norm = recorder.normalize_state(state)
            state_tensor = torch.tensor(state_norm, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                policy_logits, _ = global_net(state_tensor)
            action = int(torch.argmax(policy_logits, dim=1).item())
            recorder.update(obs, action)
            obs, reward, done, _ = test_env.step(action)
            total_reward += reward
        test_rewards.append(total_reward)
    return np.mean(test_rewards)

def a3c_train():
    global_net = A3CNet(input_dim=10, hidden_dim=128, num_actions=6)
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
