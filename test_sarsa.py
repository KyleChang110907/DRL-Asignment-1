import os
import random
import pickle
import numpy as np
import time
from environment.dynamic_env import DynamicTaxiEnv
# Adjust the import below so that get_state_self_defined is available.
from self_defined_state import get_state_self_defined 
import self_defined_state

def get_action(obs):
    """
    Given an observation (obs), convert it into the new state using get_state_self_defined,
    then load the Q-table from file and return the action with the highest Q-value.
    If the new state is not found, returns a random action.
    """
    # Convert observation into new state representation.
    new_state, _ = get_state_self_defined(obs)
    
    try:
        with open("./results_dynamic/q_table_sarsa4.pkl", "rb") as f:
            q_table = pickle.load(f)
    except FileNotFoundError:
        action = random.choice(range(6))
        
    
    if new_state in q_table:
        # print("State found in Q-table")
        action = int(np.argmax(q_table[new_state]))
        
    else:
        print("State not found in Q-table")
        action = random.choice(range(6))
    self_defined_state.last_action = action
    return action


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

        if done:
            self_defined_state.done=True
    return total_reward, step_count

def main():
    # Environment configuration parameters.
    env_config = {
        "grid_size_min": 5,
        "grid_size_max": 6,
        "fuel_limit": self_defined_state.MAX_FUEL,
        "obstacle_prob": 0.1
    }
    
    # Create the dynamic taxi environment.
    env = DynamicTaxiEnv(**env_config)
    
    num_episodes = 10
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
