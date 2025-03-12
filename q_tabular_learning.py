import numpy as np
import pickle
import random
import gym
import os
import matplotlib.pyplot as plt
from student_agent import get_action  # assuming you have this module


def get_potential(state, env):
    """
    Compute a potential based on the Manhattan distance.
    If the passenger is not in the taxi, the target is the passenger location.
    Otherwise, the target is the destination.
    """
    # Decode state: returns (taxi_row, taxi_col, passenger_location, destination)
    taxi_row, taxi_col, pass_loc, dest_idx = env.decode(state)
    # These are the four fixed locations in Taxi-v3 (R, G, Y, B)
    locations = [(0, 0), (0, 4), (4, 0), (4, 3)]
    # If the passenger is waiting (pass_loc in 0-3), target the pickup location.
    # Otherwise (pass_loc == 4) the passenger is in the taxi; target the destination.
    if pass_loc < 4:
        target = locations[pass_loc]
    else:
        target = locations[dest_idx]
    # Negative Manhattan distance as potential
    return - (abs(taxi_row - target[0]) + abs(taxi_col - target[1]))

def train_q_learning(env, num_episodes=5000, alpha=0.1, gamma=0.99, 
                       epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.01):
    """
    Train a Q-learning agent with reward shaping.
    
    - The Q-table is represented as a dictionary where each key is a state
      and the value is a numpy array of Q-values for each action.
    - After every episode the Q-table is saved to "./results/q_table.pkl".
    - Records the total reward for every episode and prints the average total reward 
      of the latest 100 episodes.
    """
    # Ensure the results directory exists.
    os.makedirs("./results", exist_ok=True)
    
    # Pre-initialize Q-table for all states in Taxi-v3
    q_table = {state: np.zeros(env.action_space.n) for state in range(env.observation_space.n)}
    
    # List to record total reward per episode
    episode_rewards = []
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        # Handle new Gym API returning (observation, info)
        if isinstance(state, tuple):
            state = state[0]
            
        done = False
        total_reward = 0
        # Compute initial potential for shaping
        potential_old = get_potential(state, env)
        
        while not done:
            # Choose an action using epsilon-greedy strategy
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))
                
            next_state, reward, done, _ , info = env.step(action)
            # Unpack next_state if needed
            if isinstance(next_state, tuple):
                next_state = next_state[0]
                
            potential_new = get_potential(next_state, env)
            # Reward shaping: add the potential difference to the original reward.
            shaped_reward = reward + gamma * potential_new - potential_old
            potential_old = potential_new
            
            # Q-learning update rule
            best_next = np.max(q_table[next_state])
            q_table[state][action] += alpha * (shaped_reward + gamma * best_next - q_table[state][action])
            
            state = next_state
            total_reward += reward  # accumulate raw reward for reporting
            
        # Record total reward for this episode
        episode_rewards.append(total_reward)
        
        # Decay epsilon over episodes
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # Save the Q-table after each episode
        with open("./results/q_table.pkl", "wb") as f:
            pickle.dump(q_table, f)
            
        # Compute and print the average total reward of the latest 100 episodes
        if len(episode_rewards) >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
        else:
            avg_reward = np.mean(episode_rewards)
        if episode % 100 == 0:  
            print(f"Episode {episode} completed. Epsilon: {epsilon:.4f} | "
                f"Avergae Reward: {np.mean(episode_rewards[-100:])} | ")
    # Plot reward history
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward History")
    plt.legend()
    plt.grid(True)
    plt.savefig("./results/q_table_reward_history.png")
    plt.show()
    
    return q_table, episode_rewards       
    


if __name__ == "__main__":
    # Create the Taxi-v3 environment
    env = gym.make("Taxi-v3")
    # Train the Q-learning agent with reward shaping
    trained_q_table = train_q_learning(env, num_episodes=10000)
    
    # Example usage of get_action:
    test_obs = env.reset()
    if isinstance(test_obs, tuple):
        test_obs = test_obs[0]
    action = get_action(test_obs)
    print(f"For observation {test_obs}, the selected action is: {action}")
