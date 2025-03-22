import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from environment.dynamic_env import DynamicTaxiEnv
from self_defined_state import StateRecorder_LargeState

# --- Hyperparameters ---
NUM_EPISODES = 3500        # Total episodes for training
MAX_STEPS = 2000              # Maximum steps per episode
GAMMA = 0.99                  # Discount factor
LR = 1e-3                   # Learning rate for optimizer
MAX_FUEL = 5000               # Fuel limit per episode
EPS_START = 1.0               # Starting value for epsilon (exploration)
EPS_END = 0.1                 # Minimum epsilon
EPS_DECAY = 3000*2000           # Decay rate for epsilon (in steps)
BATCH_SIZE = 64             # Batch size for training
REPLAY_BUFFER_SIZE = 10000  # Maximum size of replay buffer
TARGET_UPDATE_FREQ = 5000   # Steps between target network updates

# --- Q-Network Definition ---
class QNet(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=128, num_actions=6):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --- Experience Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# --- Reward Shaping Function ---
def potential(state):
    """
    Compute a potential value from the state.
    Unpack state indices:
      2: passenger_known, 3: pass_diff_r, 4: pass_diff_c,
      5: destination_known, 6: dest_diff_r, 7: dest_diff_c,
      8: passenger_on.
    Returns the negative Manhattan distance (scaled back from normalization)
    for the passenger (if not onboard) or destination (if onboard).
    """
    passenger_known = state[2]
    pass_diff_r = state[3]
    pass_diff_c = state[4]
    destination_known = state[5]
    dest_diff_r = state[6]
    dest_diff_c = state[7]
    passenger_on = state[8]
    
    if passenger_on < 0.5 and passenger_known == 1.0:
        manhattan = 10 *(abs(pass_diff_r) + abs(pass_diff_c))  # relative differences normalized by 10
        return -manhattan
    elif passenger_on >= 0.5 and destination_known == 1.0:
        manhattan = 10 * (abs(dest_diff_r) + abs(dest_diff_c))
        return -manhattan
    else:
        return 0.0

# --- Epsilon Schedule ---
def get_epsilon(step):
    """Exponential decay of epsilon."""
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1.0 * step / EPS_DECAY)

# --- DQN Training Function ---
def dqn_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = QNet(input_dim=17, hidden_dim=128, num_actions=6).to(device)
    target_net = QNet(input_dim=17, hidden_dim=128, num_actions=6).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    
    global_steps = 0
    episode_rewards = []
    test_rewards = []
    
    for ep in range(NUM_EPISODES):
        # Initialize environment and recorder for each episode.
        env = DynamicTaxiEnv(grid_size_min=5, grid_size_max=10, fuel_limit=MAX_FUEL, obstacle_prob=0.10)
        recorder = StateRecorder_LargeState(MAX_FUEL)
        obs, _ = env.reset()
        recorder.reset()
        state = recorder.get_state(obs)  # 17-dimensional state
        ep_reward = 0
        phi_old = potential(state)
        
        for t in range(MAX_STEPS):
            global_steps += 1
            epsilon = get_epsilon(global_steps)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            # ε–greedy action selection.
            if random.random() < epsilon:
                action = random.randrange(6)
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                action = int(torch.argmax(q_values, dim=1).item())
                
            # Record passenger status before taking action.
            was_passenger_on = recorder.passenger_on_taxi
            # Update recorder and take a step.
            recorder.update(obs, action)
            next_obs, reward, done, _ = env.step(action)
            now_passenger_on = recorder.passenger_on_taxi
            bonus = 0.0
            # Add bonus reward if the pickup (action 4) was successful.
            if action == 4 and (not was_passenger_on) and now_passenger_on:
                bonus = 10
                
            next_state = recorder.get_state(next_obs)
            phi_new = potential(next_state)
            shaping = GAMMA * phi_new - phi_old
            phi_old = phi_new
            r_shaped = reward + shaping + bonus
            r_shaped = r_shaped / 50.0  # Normalize the reward.
            
            ep_reward += reward
            replay_buffer.push(state, action, r_shaped, next_state, done)
            
            state = next_state
            obs = next_obs
            
            if done:
                break
            
            # Update the policy network every time we have enough experiences.
            if len(replay_buffer) >= BATCH_SIZE:
                batch = replay_buffer.sample(BATCH_SIZE)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)
                batch_state = torch.tensor(batch_state, dtype=torch.float32, device=device)
                batch_action = torch.tensor(batch_action, dtype=torch.int64, device=device).unsqueeze(1)
                batch_reward = torch.tensor(batch_reward, dtype=torch.float32, device=device).unsqueeze(1)
                batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32, device=device)
                batch_done = torch.tensor(batch_done, dtype=torch.float32, device=device).unsqueeze(1)
                
                current_q = policy_net(batch_state).gather(1, batch_action)
                with torch.no_grad():
                    max_next_q = target_net(batch_next_state).max(1)[0].unsqueeze(1)
                    target_q = batch_reward + GAMMA * max_next_q * (1 - batch_done)
                loss = F.mse_loss(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update the target network periodically.
            if global_steps % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())
        
        episode_rewards.append(ep_reward)
        # Evaluate the policy every 100 episodes.
        if (ep+1) % 100 == 0:
            avg_test_reward = evaluate_dqn(policy_net, device, num_episodes=10)
            test_rewards.append(avg_test_reward)

            # save the model
            torch.save(policy_net.state_dict(), "./results_dynamic/dqn_policy_net.pt")
            print(f"Episode {ep+1}, Average Test Reward: {avg_test_reward:.2f}, Epsilon: {epsilon:.2f}")
    
    return policy_net, episode_rewards, test_rewards

# --- Evaluation Function ---
def evaluate_dqn(policy_net, device, num_episodes=10):
    policy_net.eval()
    rewards = []
    for _ in range(num_episodes):
        env = DynamicTaxiEnv(grid_size_min=5, grid_size_max=10, fuel_limit=MAX_FUEL, obstacle_prob=0.10)
        recorder = StateRecorder_LargeState(MAX_FUEL)
        obs, _ = env.reset()
        recorder.reset()
        state = recorder.get_state(obs)
        ep_reward = 0
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
            recorder.update(obs, action)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            state = recorder.get_state(obs)
        rewards.append(ep_reward)
    policy_net.train()
    return np.mean(rewards)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net, rewards, test_rewards = dqn_train()
    os.makedirs("./results_dynamic", exist_ok=True)
    torch.save(policy_net.state_dict(), "./results_dynamic/dqn_policy_net.pt")
    np.save("./results_dynamic/dqn_rewards_history.npy", np.array(rewards))
    
    plt.figure(figsize=(10,6))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Reward History")
    plt.savefig("./results_dynamic/dqn_reward_history.png")
    plt.close()
    
    plt.figure(figsize=(10,6))
    episodes = range(0, len(test_rewards)*100, 100)
    plt.plot(episodes, test_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Test Reward History")
    plt.savefig("./results_dynamic/dqn_test_reward_history.png")
    plt.close()
