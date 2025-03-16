import os
import random
import time
import numpy as np
import torch
from env.dynamic_env import DynamicTaxiEnv  # adjust import as needed
from DQN import DQN

def run_agent(model_file, env_config, render=False):
    """
    Runs a single episode using the trained DQN network.
    
    Parameters:
      model_file: Path to the file storing the trained DQN network weights.
      env_config: Dictionary with environment configuration parameters.
      render: If True, prints the environment's rendered text at each step.
    
    Returns:
      total_reward: The cumulative reward achieved in the episode.
    """
    # Set device and load the DQN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(input_dim=10, output_dim=6).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    print("DQN model loaded successfully from", model_file)
    
    # Create the environment (each episode gets a random grid size)
    env = DynamicTaxiEnv(**env_config)
    obs, _ = env.reset()  # obs is a tuple with 10 elements
    total_reward = 0
    done = False
    step_count = 0

    if render:
        print(env.render())
        time.sleep(0.5)

    while not done:
        # Convert observation to tensor
        state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_vals = model(state_tensor)
        # Choose action with highest Q-value
        action = int(torch.argmax(q_vals, dim=1).item())
        
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        step_count += 1
        obs = next_obs
        
        if render:
            print(env.render())
            time.sleep(0.2)

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward

if __name__ == "__main__":
    env_config = {
        "grid_size_min": 5,
        "grid_size_max": 10,
        "fuel_limit": 5000,
        "obstacle_prob": 0.1
    }
    # Adjust the path to your trained DQN model as needed.
    run_agent(model_file="./results_dynamic/dqn_policy_net.pt", env_config=env_config, render=True)
