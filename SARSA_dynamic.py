import os
import random
import pickle
import numpy as np
from env.dynamic_env import DynamicTaxiEnv
from self_defined_state import get_state_self_defined, MAX_FUEL

NUM_EPISODES = 150000

# ---------------------------
# Potential Function for Reward Shaping
# ---------------------------
def potential(state, env):
    """
    Compute a potential based on the new state representation:
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
    return -distance / (env.grid_size - 1)


# ---------------------------
# SARSA Training with Reward Shaping Using New State Representation
# ---------------------------
def train_dynamic_taxi_sarsa(env, num_episodes=1000, alpha=0.1, gamma=0.99, 
                             epsilon=1.0, epsilon_decay=0.99999, min_epsilon=0.1):
    """
    Train a SARSA agent on the dynamic grid-world taxi with random grid sizes.
    Uses potential-based reward shaping.
    
    New state representation for training:
      (pass_row_diff, pass_col_diff, dest_row_diff, dest_col_diff, picked, obs_n, obs_s, obs_e, obs_w, fuel)
    """
    q_table = {}
    rewards_history = []

    def get_q(state):
        # Use the state tuple as key.
        if state not in q_table:
            q_table[state] = np.zeros(6)  # 6 possible actions
        return q_table[state]

    for episode in range(num_episodes):
        # print(f'=====================Episode: {episode}================')
        # print(f'=====================Step: 0================')
        global last_action
        # print("last_action: ", last_action)
        obs, _ = env.reset()  # Get original state from the environment.
        state, other_state = get_state_self_defined(obs)
        done = False
        total_reward = 0
        step = 1
        phi_old = potential(state, env)
        
        # Choose initial action using epsilon-greedy.
        if random.random() < epsilon:
            action = random.choice(range(6))
        else:
            action = int(np.argmax(get_q(state)))
        
        last_action = action
        # print("action: ", action)
        
        while not done:
            step+=1
            # print(f'=====================Episode: {episode}================')
            # print(f'=====================Step: {step}================')
            # print("last_action: ", last_action)
            next_obs, reward, done, _ = env.step(action)
            next_state, next_other_state = get_state_self_defined(next_obs)
            phi_new = potential(next_state, env)
            shaping = (gamma * phi_new - phi_old)*10
            # print("shaping: ", shaping)
            # print(f'phi_new: {phi_new}, phi_old: {phi_old}')
            # Special shaping modifications.
            # if state[2] == 0 and next_state[2] == 1 and action == 4:
            if state[2] == 0 and next_state[2] == 1:
                shaping = 25
                # print("Picked up passenger! Reward:", reward)
            # elif action==5 and other_state[2]==0 and other_state[3]==0:
            #     shaping = 10000
            #     # print("Dropped off passenger! Reward:", reward)
            # elif action == 5 and (next_state[2] != 0 or next_state[3] != 0):
            #     # print("Dropped off passenger at wrong location! Reward:", reward)
            #     shaping = -10000
            
            # if not picked and do dropoff
            # if state[2] == 0 and action == 5:
            #     shaping = -2000
                # print("Dropped off passenger without picking Passenger! Reward:", reward)
            # if picking at the wrong location
            if state[2] == 0 and other_state[0] != 0 and other_state[1] != 0 and action == 4:
                shaping = -15
                # print("Picked up passenger at wrong location! Reward:", reward)
            
            phi_old = phi_new
            r_shaped = reward + shaping

            # Choose next action using epsilon-greedy (if not terminal).
            if not done:
                if random.random() < epsilon:
                    next_action = random.choice(range(6))
                else:
                    next_action = int(np.argmax(get_q(next_state)))
            else:
                next_action = 0  # arbitrary

            # SARSA update.
            get_q(state)[action] += alpha * (r_shaped + gamma * get_q(next_state)[next_action] - get_q(state)[action])
            state = next_state
            other_state = next_other_state
            action = next_action
            
            last_action = action
            # print("action: ", action)
            total_reward += reward

            # if done:
            #     if reward == 49.9:
            #         print("==============Dropped off passenger correctly==========")

            global Fuel
            # print("Fuel: ", Fuel)
            # print("Step: ", step)
            if Fuel != step:
                if step == MAX_FUEL+1:
                    continue
                else:
                    print("Fuel not equal to step")
                    print(f"Fuel: {Fuel}, Step: {step}")
                
                raise ValueError("Fuel not equal to step")

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
    env = DynamicTaxiEnv(grid_size_min=5, grid_size_max=10, fuel_limit=MAX_FUEL, obstacle_prob=0.1)
    q_table, rewards_history = train_dynamic_taxi_sarsa(env, num_episodes=NUM_EPISODES)
    os.makedirs("./results_dynamic", exist_ok=True)
    with open("./results_dynamic/q_table_sarsa.pkl", "wb") as f:
        pickle.dump(q_table, f)
    np.save("./results_dynamic/rewards_history_sarsa.npy", rewards_history)

    # Plot reward history.
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("SARSA Reward History")
    plt.legend()
    plt.grid(True)
    plt.savefig("./results_dynamic/q_table_sarsa_reward_history.png")
    plt.close()
