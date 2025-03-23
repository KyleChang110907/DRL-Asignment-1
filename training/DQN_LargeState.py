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
from self_defined_state import StateRecorder_Enhanced

sav_dir = "./results_dynamic/dqn_policy_net_3.pt"
# --- Hyperparameters ---
NUM_EPISODES = 20000        # Total episodes for training
MAX_STEPS = 1000              # Maximum steps per episode
GAMMA = 0.99                  # Discount factor
LR = 1e-3                   # Learning rate for optimizer
MAX_FUEL = 5000               # Fuel limit per episode
EPS_START = 1.0               # Starting value for epsilon (exploration)
EPS_END = 0.1                 # Minimum epsilon
EPS_DECAY = 13000*1000           # Decay rate for epsilon (in steps)
BATCH_SIZE = 128             # Batch size for training
REPLAY_BUFFER_SIZE = 20000  # Maximum size of replay buffer
TARGET_UPDATE_FREQ = 1000   # Steps between target network updates

# --- Q-Network Definition ---
class QNet(nn.Module):
    def __init__(self, input_dim=27, hidden_dim=64, num_actions=6):
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
    Compute a potential value from the enhanced state.
    
    The state vector is assumed to have 27 dimensions:
      - Indices 0-1: Taxi's normalized position.
      - Indices 2-9: Station positions (4 stations, each 2 dims).
      - Indices 10-13: Passenger one-hot indicator (which station is passenger).
      - Indices 14-17: Destination one-hot indicator (which station is destination).
      - Index 18: Passenger on taxi flag (1 if onboard, 0 otherwise).
      - Remaining indices: fuel, last action, obstacles, raw look flags.
    
    If the passenger is not onboard and a passenger station is known, the potential is defined as 
    the negative Manhattan distance from the taxi to the passenger's station multiplied by 15.
    If the passenger is onboard and a destination is known, the potential is defined similarly.
    If the corresponding indicator is not set, the potential is 0.
    """
    # Taxi's normalized absolute position.
    taxi_r = state[0]
    taxi_c = state[1]
    
    # Station positions: indices 2-9.
    stations = [
        (state[2], state[3]),
        (state[4], state[5]),
        (state[6], state[7]),
        (state[8], state[9])
    ]
    
    # Passenger one-hot indicator: indices 10-13.
    passenger_onehot = list(state[10:14])
    # Destination one-hot indicator: indices 14-17.
    destination_onehot = list(state[14:18])
    
    # Passenger on taxi flag: index 18.
    passenger_on = state[18]
    
    scale = 15.0  # Scaling factor to amplify the Manhattan distance
    
    if passenger_on < 0.5:
        # Use passenger station as target.
        if sum(passenger_onehot) < 0.5:
            # If no passenger station is known, return 0 potential.
            return 0.0
        # Get the index corresponding to the maximum one-hot value.
        passenger_index = passenger_onehot.index(max(passenger_onehot))
        target_r, target_c = stations[passenger_index]
        # Manhattan distance in normalized space.
        manhattan = abs(taxi_r - target_r) + abs(taxi_c - target_c)
        return -scale * manhattan
    else:
        # Passenger is onboard; use destination station.
        if sum(destination_onehot) < 0.5:
            return 0.0
        destination_index = destination_onehot.index(max(destination_onehot))
        target_r, target_c = stations[destination_index]
        manhattan = abs(taxi_r - target_r) + abs(taxi_c - target_c)
        return -scale * manhattan

# --- Epsilon Schedule ---
def get_epsilon(step):
    """Exponential decay of epsilon."""
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1.0 * step / EPS_DECAY)

# --- DQN Training Function ---
def dqn_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = QNet(input_dim=26, hidden_dim=128, num_actions=6).to(device)
    target_net = QNet(input_dim=26, hidden_dim=128, num_actions=6).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    
    global_steps = 0
    episode_rewards = []
    test_rewards = []
    losses = []
    
    for ep in range(NUM_EPISODES):
        # Initialize environment and recorder for each episode.
        env = DynamicTaxiEnv(grid_size_min=5, grid_size_max=10, fuel_limit=MAX_FUEL, obstacle_prob=0.10)
        recorder = StateRecorder_Enhanced(MAX_FUEL)
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

            if reward == -5.1:
                # print('crush into obstacle')
                bonus = -10
                
            next_state = recorder.get_state(next_obs)
            phi_new = potential(next_state)
            shaping = GAMMA * phi_new - phi_old
            phi_old = phi_new
            r_shaped = reward + shaping + bonus

            if bonus > 0 or reward > 0:
                shaping = 0 
                r_shaped = reward + bonus + shaping
                # print(f'state: {state}, action: {action}')
                # print(f'reward: {reward}, shaping: {shaping}, bonus: {bonus}, r_shaped: {r_shaped}')
            
            # print(f'state: {state}, action: {action}')
            # print(f'reward: {reward}, shaping: {shaping}, bonus: {bonus}, r_shaped: {r_shaped}')
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
                losses.append(loss.item())
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
            torch.save(policy_net.state_dict(), sav_dir )
            print(f"Episode {ep+1}, Average Test Reward: {avg_test_reward:.2f}, average loss:{sum(losses[-100:])/100:.4f}, Epsilon: {epsilon:.2f}")
    
    return policy_net, episode_rewards, test_rewards, losses

# --- Evaluation Function ---
def evaluate_dqn(policy_net, device, num_episodes=10):
    policy_net.eval()
    rewards = []
    for _ in range(num_episodes):
        env = DynamicTaxiEnv(grid_size_min=5, grid_size_max=10, fuel_limit=MAX_FUEL, obstacle_prob=0.10)
        recorder = StateRecorder_Enhanced(MAX_FUEL)
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
    policy_net, rewards, test_rewards, losses = dqn_train()
    os.makedirs("./results_dynamic", exist_ok=True)
    torch.save(policy_net.state_dict(), sav_dir )
    np.save("./results_dynamic/dqn_rewards_history_3.npy", np.array(rewards))
    
    plt.figure(figsize=(10,6))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Reward History")
    plt.savefig("./results_dynamic/dqn_reward_history_3.png")
    plt.close()
    
    plt.figure(figsize=(10,6))
    episodes = range(0, len(test_rewards)*100, 100)
    plt.plot(episodes, test_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Test Reward History")
    plt.savefig("./results_dynamic/dqn_test_reward_history_3.png")
    plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("DQN Loss History")
    plt.savefig("./results_dynamic/dqn_loss_history_3.png")
    plt.close()
