import os
import random
import pickle
import numpy as np
from environment.dynamic_env import DynamicTaxiEnv
from self_defined_state import StateRecorder  # import the class from your file

NUM_EPISODES = 250000
MAX_FUEL = 5000

def potential(state, env):
    # Using the new state produced by StateRecorder, we define potential as the negative
    # of the sum of the absolute differences in the "going" direction.
    # Our new state is: (going_row_diff, going_col_diff, picked, last_action_norm, visited_code_norm, fuel_norm, obs_n, obs_s, obs_e, obs_w)
    return - (abs(state[0]) + abs(state[1])) / (env.grid_size - 1)

def train_dynamic_taxi_sarsa(env, num_episodes=1000, alpha=0.1, gamma=0.99,
                             epsilon=1.0, epsilon_decay=0.99997, min_epsilon=0.1):
    """
    Train a SARSA agent on the dynamic taxi environment using the StateRecorder for state representation.
    """
    q_table = {}
    rewards_history = []
    
    # Create a global state recorder instance.
    recorder = StateRecorder(MAX_FUEL)
    
    def get_q(state):
        if state not in q_table:
            q_table[state] = np.zeros(6)  # 6 actions
        return q_table[state]
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        recorder.reset()  # Reset the recorder at the start of the episode.
        state, other_state = recorder.get_state(obs)
        done = False
        total_reward = 0
        step = 1
        phi_old = potential(state, env)
        
        # Choose initial action epsilon-greedy.
        if random.random() < epsilon:
            action = random.choice(range(6))
        else:
            action = int(np.argmax(get_q(state)))
        recorder.update(obs, action)
        
        while not done:
            step += 1
            next_obs, reward, done, _ = env.step(action)
            recorder.update(next_obs, action)  # update recorder with current observation and last action.
            next_state, next_other_state = recorder.get_state(next_obs)
            phi_new = potential(next_state, env)
            shaping = (gamma * phi_new - phi_old) * 10
            phi_old = phi_new

            # chnaging shpaing if the action is pickup or dropoff
            if state[2] == 0 and next_state[2] == 1 and action == 4:
                shaping = 100
                # print("Picked up passenger! Reward:", reward)
            elif action==4 and state[2]!=1:
                shaping = -10
            elif action==5 and state[2]==0:
                shaping = -100
                # print("Picked up passenger at wrong location! Reward:", reward)
            elif action==5 and other_state[2]==0 and other_state[3]==0 and state[2] == 1:
                shaping = 500
                # print("********Dropped off passenger correctly! Reward:", reward)
            elif action == 5 :
                # print("Dropped off passenger at wrong location! Reward:", reward)
                shaping = -500
            
            # (Optional) You can add special shaping modifications here if needed.
            if not done:
                if random.random() < epsilon:
                    next_action = random.choice(range(6))
                else:
                    next_action = int(np.argmax(get_q(next_state)))
            else:
                next_action = 0  # arbitrary
            
            # SARSA update.
            get_q(state)[action] += alpha * (reward + shaping + gamma * get_q(next_state)[next_action] - get_q(state)[action])
            state = next_state
            other_state = next_other_state
            action = next_action
            total_reward += reward
        
        rewards_history.append(total_reward)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode+1}: Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
    
    return q_table, rewards_history

if __name__ == "__main__":
    env = DynamicTaxiEnv(grid_size_min=5, grid_size_max=10, fuel_limit=MAX_FUEL, obstacle_prob=0.10)
    q_table, rewards_history = train_dynamic_taxi_sarsa(env, num_episodes=NUM_EPISODES)
    os.makedirs("./results_dynamic", exist_ok=True)
    with open("./results_dynamic/q_table_sarsa4.pkl", "wb") as f:
        pickle.dump(q_table, f)
    np.save("./results_dynamic/rewards_history_sarsa.npy", rewards_history)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("SARSA Reward History")
    plt.legend()
    plt.grid(True)
    plt.savefig("./results_dynamic/q_table_sarsa_reward_history4.png")
    plt.close()
