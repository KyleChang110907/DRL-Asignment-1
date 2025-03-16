import os
import pickle
import random
import time
import numpy as np
from environment.dynamic_env import DynamicTaxiEnv  # adjust import as needed

def run_agent(q_table_file, env_config, render=False):
    """
    Runs a single episode using the trained Q-table.
    
    Parameters:
      q_table_file: Path to the pickle file storing the trained Q-table.
      env_config: Dictionary with environment configuration parameters.
      render: If True, prints the environment's rendered text at each step.
    
    Returns:
      total_reward: The cumulative reward achieved in the episode.
    """
    # Load the trained Q-table
    try:
        with open(q_table_file, "rb") as f:
            q_table = pickle.load(f)
        print("Q-table loaded successfully from", q_table_file)
    except Exception as e:
        print("Error loading Q-table:", e)
        return

    # Create the environment (each episode gets a random grid size)
    env = DynamicTaxiEnv(**env_config)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0

    if render:
        print(env.render())
        time.sleep(0.5)

    while not done:
        # Get action from the Q-table (state is assumed to be a tuple)
        if obs in q_table:
            action = int(np.argmax(q_table[obs]))
        else:
            action = random.choice(range(6))
        
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        step_count += 1
        obs = next_obs
        
        if render:
            # Print the rendered grid (text output)
            print(env.render())
            time.sleep(0.2)

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward

if __name__ == "__main__":
    env_config = {
        "grid_size_min": 5,
        "grid_size_max": 10,
        "fuel_limit": 100,
        "obstacle_prob": 0.1
    }
    # Adjust the path to your trained Q-table as needed.
    run_agent(q_table_file="./results_dynamic/q_table.pkl", env_config=env_config, render=True)
