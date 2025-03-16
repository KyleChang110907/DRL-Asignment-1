import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from env.dynamic_env import DynamicTaxiEnv  # Adjust path as needed

# ---------------------------
# Policy Network Definition
# ---------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_actions=6):
        """
        input_size: Dimension of the state vector (should be 10)
        hidden_size: Number of hidden units
        num_actions: Number of discrete actions (6 for taxi)
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

# ---------------------------
# Helper Function: Discount Rewards
# ---------------------------
def discount_rewards(rewards, gamma):
    """Compute the discounted returns for an episode."""
    discounted = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = rewards[t] + gamma * running_add
        discounted[t] = running_add
    return discounted

# ---------------------------
# Policy Gradient (REINFORCE) Training Function
# ---------------------------
def train_policy_gradient(env, num_episodes=1000, gamma=0.99, learning_rate=1e-3, render=False):
    """
    Trains a policy using the REINFORCE algorithm.
    
    Parameters:
      env: an instance of DynamicTaxiEnv.
      num_episodes: total episodes for training.
      gamma: discount factor.
      learning_rate: learning rate for Adam optimizer.
      render: if True, prints env.render() each step.
      
    Returns:
      policy_net: the trained policy network.
      rewards_history: list of total rewards per episode.
    """
    # Initialize the policy network and optimizer.
    policy_net = PolicyNetwork(input_size=10, hidden_size=128, num_actions=6)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    rewards_history = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()  # new state representation as tuple of length 10
        log_probs = []
        rewards = []
        done = False
        total_reward = 0
        
        while not done:
            # Convert state to a torch tensor.
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probs = policy_net(state_tensor)
            # Sample an action from the categorical distribution.
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            log_probs.append(log_prob)
            
            next_state, reward, done, _ = env.step(action.item())
            rewards.append(reward)
            total_reward += reward
            state = next_state
            
            if render:
                print(env.render())
        
        rewards_history.append(total_reward)
        # Compute discounted rewards.
        discounted_rewards = discount_rewards(rewards, gamma)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        # Normalize discounted rewards.
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        # Compute policy loss.
        policy_loss = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * Gt)
        policy_loss = torch.cat(policy_loss).sum()
        
        # Backpropagation.
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if (episode+1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode+1}: Total Reward: {total_reward:.2f}, Avg Reward (last 100): {avg_reward:.2f}")
    
    return policy_net, rewards_history

# ---------------------------
# Main: Training and Saving Results
# ---------------------------
if __name__ == "__main__":
    # Create the dynamic taxi environment (each episode has a random grid size between 5x5 and 15x15).
    env = DynamicTaxiEnv(grid_size_min=5, grid_size_max=10, fuel_limit=5000, obstacle_prob=0.2)
    num_episodes = 10  # adjust as needed
    policy_net, rewards_history = train_policy_gradient(env, num_episodes=num_episodes, gamma=0.99, learning_rate=1e-3, render=False)
    
    # Save the trained policy network and reward history.
    os.makedirs("./results_dynamic", exist_ok=True)
    torch.save(policy_net.state_dict(), "./results_dynamic/policy_net.pt")
    np.save("./results_dynamic/pg_rewards_history.npy", rewards_history)
    
    # Plot reward history.
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Policy Gradient Reward History")
    plt.legend()
    plt.grid(True)
    plt.savefig("./results_dynamic/pg_reward_history.png")
    plt.close()
