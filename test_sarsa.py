import os
import random
import pickle
import numpy as np
import time
from env.dynamic_env import DynamicTaxiEnv
# Adjust the import below so that get_state_self_defined is available.
from SARSA_dynamic import get_state_self_defined 
def get_action(obs):
    """
    Given an observation (obs), convert it into the new state using get_state_self_defined,
    then load the Q-table from file and return the action with the highest Q-value.
    If the new state is not found, returns a random action.
    """
    # Convert observation into new state representation.
    new_state, _ = get_state_self_defined(obs)
    
    try:
        with open("./results_dynamic/q_table_sarsa.pkl", "rb") as f:
            q_table = pickle.load(f)
    except FileNotFoundError:
        return random.choice(range(6))
    
    if new_state in q_table:
        print("State found in Q-table")
        return int(np.argmax(q_table[new_state]))
    else:
        return random.choice(range(6))

def run_episode(env, render=False):
    """
    Runs one episode of the environment using get_action to select actions.
    
    Returns the total reward and the number of steps taken.
    """
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    step_count = 0
    while not done:
        action = get_action(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        step_count += 1
        if render:
            print(env.render())
            time.sleep(0.2)
    return total_reward, step_count

def main():
    # Environment configuration parameters.
    env_config = {
        "grid_size_min": 5,
        "grid_size_max": 10,
        "fuel_limit": 5000,
        "obstacle_prob": 0.1
    }
    
    # Create the dynamic taxi environment.
    env = DynamicTaxiEnv(**env_config)
    
    num_episodes = 1
    rewards = []
    steps = []
    
    for ep in range(num_episodes):
        print(f"================= Episode {ep+1} =================")
        total_reward, step_count = run_episode(env, render=True)
        print(f"Episode {ep+1} finished: Total Reward = {total_reward:.2f}, Steps = {step_count}")
        rewards.append(total_reward)
        steps.append(step_count)
    
    print("Testing finished.")
    print("Episode Rewards:", rewards)
    print("Episode Steps:", steps)

if __name__ == "__main__":
    main()
