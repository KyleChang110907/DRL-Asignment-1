import os
import random
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
from environment.dynamic_env import DynamicTaxiEnv
from self_defined_state import StateRecorder  # Your class-based state recorder
from student_agent import get_action_sarsa as get_action  # Your agent function

# Testing configuration
NUM_TEST_EPISODES = 10
MAX_FUEL = 5000


def test_agent(env, num_episodes):
    """
    Run the SARSA agent (using the trained Q-table) for a number of episodes.
    Returns a list of total rewards per episode.
    """
    episode_rewards = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = get_action(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}")
    
    avg_reward = np.mean(episode_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")
    return episode_rewards

if __name__ == "__main__":
    # Create the environment.
    env = DynamicTaxiEnv(grid_size_min=5, grid_size_max=10, fuel_limit=MAX_FUEL, obstacle_prob=0.10)
    
    # Load the trained Q-table for SARSA.
    q_table_file = "./results_dynamic/q_table_sarsa4.pkl"
    try:
        with open(q_table_file, "rb") as f:
            q_table = pickle.load(f)
        print("Q-table loaded successfully from:", q_table_file)
    except Exception as e:
        print("Error loading Q-table:", e)
        exit(1)
    
    # Run testing episodes.
    rewards = test_agent(env, NUM_TEST_EPISODES)
    
    # Plot reward history.
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, marker='o', label="Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("SARSA Agent Testing Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("./results_dynamic/sarsa_testing_reward.png")
    plt.close()
