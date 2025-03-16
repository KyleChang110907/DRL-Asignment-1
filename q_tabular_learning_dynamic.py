import os
import random
import pickle
import numpy as np
from env.dynamic_env import DynamicTaxiEnv

# ---------------------------
# Reward Shaping Strategy
# ---------------------------
def potential(state, env):
    """
    Compute a potential function based on the new state representation:
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
    return -distance * 10
    return -distance / (env.grid_size - 1)

# ---------------------------
# Q-Learning Training with Reward Shaping
# ---------------------------
def train_dynamic_taxi(env, num_episodes=1000, alpha=0.1, gamma=0.99, 
                         epsilon=1.0, epsilon_decay=0.9999, min_epsilon=0.1):
    """
    Train a Q-learning agent on the dynamic grid-world taxi with random grid sizes.
    Uses potential-based reward shaping.
    
    New state representation:
      (pass_row_diff, pass_col_diff, dest_row_diff, dest_col_diff, picked, obs_n, obs_s, obs_e, obs_w, fuel)
    """
    q_table = {}
    rewards_history = []

    def get_q(state):
        # Use the state tuple as the key.
        if state not in q_table:
            q_table[state] = np.zeros(6)  # 6 possible actions
        return q_table[state]

    for episode in range(num_episodes):
        state, _ = env.reset()  # state now is the new representation.
        done = False
        total_reward = 0
        phi_old = potential(state, env)
        while not done:
            if random.random() < epsilon:
                action = random.choice(range(6))
            else:
                q = get_q(state)
                action = int(np.argmax(q))
            next_state, reward, done, _ = env.step(action)
            phi_new = potential(next_state, env)
            shaping = gamma * phi_new - phi_old

            # If passenger has just been picked up (i.e. picked flag changes from 0 to 1), override shaping.
            if state[4] == 0 and next_state[4] == 1:
                shaping = 10000
                # print("Picked up passenger! Reward: ", reward)
            # elif properly dropped off passenger
            elif state[4] == 1 and next_state[4] == 0 and state[2] == 0 and state[3] == 0:
                shaping = 20000
                print("Dropped off passenger! Reward: ", reward)
            # elif not properly dropped off passenger
            elif state[4] == 1 and next_state[4] == 0 and done:
                shaping = -2000

            phi_old = phi_new
            r_shaped = reward + shaping
            best_next = np.max(get_q(next_state))
            get_q(state)[action] += alpha * (r_shaped + gamma * best_next - get_q(state)[action])
            state = next_state
            total_reward += reward
        rewards_history.append(total_reward)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode+1}: Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
    return q_table, rewards_history

# ---------------------------
# Main: Training and Saving Results
# ---------------------------
if __name__ == "__main__":
    # Create environment with random grid size per episode.
    env = DynamicTaxiEnv(grid_size_min=5, grid_size_max=10, fuel_limit=5000, obstacle_prob=0.2)
    q_table, rewards_history = train_dynamic_taxi(env, num_episodes=20000)
    os.makedirs("./results_dynamic", exist_ok=True)
    with open("./results_dynamic/q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)
    np.save("./results_dynamic/rewards_history.npy", rewards_history)

    # plot reward history
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward History")
    plt.legend()
    plt.grid(True)
    plt.savefig("./results_dynamic/q_table_reward_history.png")
    plt.close()

    
