import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
from env.dynamic_env import DynamicTaxiEnv

# ---------------------------
# Potential Function for Reward Shaping
# ---------------------------
def potential(state, env):
    """
    Compute a potential based on the new state representation:
      state = (pass_row_diff, pass_col_diff, dest_row_diff, dest_col_diff, picked, obs_n, obs_s, obs_e, obs_w, fuel)
    
    - If the passenger is not picked (picked==0), target is the passenger:
         potential = -(pass_row_diff + pass_col_diff) / (env.grid_size - 1)
    - If the passenger is picked (picked==1), target is the destination:
         potential = -(dest_row_diff + dest_col_diff) / (env.grid_size - 1)
    """
    picked = state[4]
    if picked == 0:
        distance = state[0] + state[1]
    else:
        distance = state[2] + state[3]
    # Use normalized potential; remove the unreachable second return.
    return -distance 

# ---------------------------
# DQN Network Definition
# ---------------------------
class DQN(nn.Module):
    def __init__(self, input_dim=10, output_dim=6):
        """
        input_dim: dimension of the state vector (10 for our new representation)
        output_dim: number of actions (6 for taxi)
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ---------------------------
# Replay Buffer
# ---------------------------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# ---------------------------
# Hyperparameters
# ---------------------------
BATCH_SIZE = 128
REPLAY_BUFFER_CAPACITY = 10000
TARGET_UPDATE_FREQ = 1000  # in steps
NUM_EPISODES = 5000
GAMMA = 0.99
LEARNING_RATE = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.9999
SAVE_EVERY = 100  # Save the model every 100 episodes

# ---------------------------
# DQN Training Function
# ---------------------------
def train_dqn(env):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Training DQN Agent on Dynamic Taxi Environment")
    print("Using device:", device)

    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
    
    epsilon = EPSILON_START
    total_steps = 0
    rewards_history = []
    
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()  # state is in the new representation (tuple of length 10)
        done = False
        total_reward = 0
        
        # Compute initial potential.
        phi_old = potential(state, env)
        
        while not done:
            total_steps += 1
            # Epsilon-greedy action selection.
            if random.random() < epsilon:
                action = random.randrange(6)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q_vals = policy_net(state_tensor)
                action = int(torch.argmax(q_vals, dim=1).item())
            
            next_state, reward, done, _ = env.step(action)
            
            # Compute potential-based shaping reward.
            phi_new = potential(next_state, env)
            shaping = GAMMA * phi_new - phi_old
            phi_old = phi_new

            # Additional shaping modifications.
            if state[4] == 0 and next_state[4] == 1:
                shaping = 10000
            if state[4] == 1 and state[2] == 0 and state[3] == 0 and action == 5:
                shaping = 100000
                print("Dropped off passenger at destination! Reward:", reward)
            if state[4] == 1 and (state[2] != 0 or state[3] != 0) and action == 5:
                shaping = -10000
                print("Dropped off passenger at wrong location! Reward:", reward)
        
            r_shaped = reward + shaping

            # Store transition in replay buffer.
            replay_buffer.push(state, action, r_shaped, next_state, done)
            state = next_state
            total_reward += reward  # accumulate original reward
            
            # If enough samples, perform DQN update.
            if len(replay_buffer) >= BATCH_SIZE:
                states, actions, rewards_batch, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                states = torch.tensor(states, dtype=torch.float32, device=device)
                actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
                rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32, device=device)
                next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones = torch.tensor(dones, dtype=torch.float32, device=device)
                
                current_q = policy_net(states).gather(1, actions).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                    target_q = rewards_batch + GAMMA * next_q * (1 - dones)
                
                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update target network periodically.
            if total_steps % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())
        
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        rewards_history.append(total_reward)
        print(f"Episode {episode+1}: Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
        if (episode+1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode+1}: Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
            # Save the current policy network during training.
            torch.save(policy_net.state_dict(), f"./results_dynamic/dqn_policy_net_episode_{episode+1}.pt")
    
    return policy_net, rewards_history

# ---------------------------
# Main: Train and Save Final Results
# ---------------------------
if __name__ == "__main__":
    # Create environment with random grid size per episode.
    env = DynamicTaxiEnv(grid_size_min=5, grid_size_max=10, fuel_limit=5000, obstacle_prob=0.15)
    policy_net, rewards_history = train_dqn(env)
    os.makedirs("./results_dynamic", exist_ok=True)
    # Save final model.
    torch.save(policy_net.state_dict(), "./results_dynamic/dqn_policy_net_final.pt")
    np.save("./results_dynamic/dqn_rewards_history.npy", rewards_history)
    
    # Plot reward history.
    plt.figure(figsize=(10,6))
    plt.plot(rewards_history, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Reward History")
    plt.legend()
    plt.grid(True)
    plt.savefig("./results_dynamic/dqn_reward_history.png")
    plt.close()
