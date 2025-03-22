import os
import time
import numpy as np
from environment.dynamic_env import DynamicTaxiEnv
from student_agent import get_action  # Adjust this import to where your get_action is defined

NUM_TEST_EPISODES = 10

def test_agent(env, num_episodes, render=False):
    episode_rewards = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        step = 0
        while not done:
            if render:
                print(env.render())
                time.sleep(0.5)
            action = get_action(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            step += 1
        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}: Total Reward = {total_reward:.2f} in {step} steps")
    
    avg_reward = np.mean(episode_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")
    return episode_rewards

if __name__ == "__main__":
    # Create environment with desired configuration.
    env = DynamicTaxiEnv(grid_size_min=5, grid_size_max=10, fuel_limit=5000, obstacle_prob=0.1)
    rewards = test_agent(env, NUM_TEST_EPISODES, True)
