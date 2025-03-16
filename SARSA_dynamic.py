import os
import random
import pickle
import numpy as np
from env.dynamic_env import DynamicTaxiEnv

MAX_FUEL = 5000
NUM_EPISODES = 150000
# globally storage
# ---------------------------
Destination_Location = [-1, -1]
Passenger_Location = [-1, -1]
Passenger_on_Taxi = False
Been_to_Stations = [False, False, False, False]
last_action = -1
Done_or_not = False
Fuel = 0

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
# New State Conversion Function for Training
# ---------------------------
def get_state_self_defined(obs):
    """
    Convert the original environment state (obs) into a new state representation for training.
    
    Original obs is a 16-tuple:
      (taxi_row, taxi_col,
       station0_row, station0_col, station1_row, station1_col,
       station2_row, station2_col, station3_row, station3_col,
       obstacle_north, obstacle_south, obstacle_east, obstacle_west,
       passenger_look, destination_look)
    
    New state (10-tuple):
      (pass_row_diff, pass_col_diff, dest_row_diff, dest_col_diff, picked,
       obs_n, obs_s, obs_e, obs_w, fuel)
       
    Assumptions:
      - If passenger_look (obs[14]) is 1, assume the taxi sees the passenger:
          set pass_row_diff = 0 and pass_col_diff = 0.
        Otherwise, assume the passenger is at the station (from indices 2–9) that is closest to the taxi.
      - If destination_look (obs[15]) is 1, assume the taxi sees the destination:
          set dest_row_diff = 0 and dest_col_diff = 0.
        Otherwise, assume the destination is at the station that is farthest from the taxi.
      - Define picked = 0 if passenger_look is 1 (not picked) and 1 otherwise.
      - Since fuel is not provided in obs, set fuel = 0.
    """
    taxi_r, taxi_c = obs[0], obs[1]
    # Station coordinates
    s0 = (obs[2], obs[3])
    s1 = (obs[4], obs[5])
    s2 = (obs[6], obs[7])
    s3 = (obs[8], obs[9])
    stations = [s0, s1, s2, s3]
    
    # Obstacle indicators.
    obs_n = obs[10]
    obs_s = obs[11]
    obs_e = obs[12]
    obs_w = obs[13]
    
    passenger_look = obs[14]
    destination_look = obs[15]

    
    global Passenger_Location
    global Destination_Location
    global Passenger_on_Taxi
    global Been_to_Stations
    global Done_or_not
    global last_action
    global Fuel

    # If Done_or_not is True, reset global storage.
    if Done_or_not:
        reset_global_storage()
    
    # print("Passenger_Location: ", Passenger_Location)
    # print("Destination_Location: ", Destination_Location)
    # print("Passenger_on_Taxi: ", Passenger_on_Taxi)
    # print("Been_to_Stations: ", Been_to_Stations)
    # print("Done_or_not: ", Done_or_not)
    # print("last_action: ", last_action)
    # print("Fuel: ", Fuel)

    # print(f'Current Taxi Position: ({taxi_r}, {taxi_c})')
    # print("passenger_look", passenger_look)
    # print('stations: ', stations)
    
    # Find passenger station
    # Know Passenger station if the passenger_look is 1 and taxi at one of the stations.
    if Passenger_Location == [-1,-1] and passenger_look==1 and ((taxi_r, taxi_c) in stations):
            Passenger_Location = [taxi_r, taxi_c]
            # print("Found the Passenger station")
            # print(f'statios: {stations}')

    # Passenger on taxi if already known passenger places and taxi and passenger are at the same station and last action was pickup.
    elif not Passenger_on_Taxi and [taxi_r, taxi_c] == Passenger_Location and last_action == 4:
        Passenger_on_Taxi = True
        # print("Passenger are picked up")

    # If passenger is on taxi, reset passenger location. The passenger would be the same location at the taxi.
    if Passenger_on_Taxi :
        Passenger_Location = [taxi_r, taxi_c]
        # If passenger is on taxi and the last action was dropoff, set passenger not on the taxi.
        if last_action == 5 and Destination_Location == [taxi_r, taxi_c]:
            Done_or_not = True
            # print("Passenger are dropped off")
        elif last_action == 5 and Destination_Location != [taxi_r, taxi_c]:
            Passenger_on_Taxi = False
            # print("Passenger are dropped off at wrong location")
    
    # Find destination station
    # If the destination station is unkown and destination_look is 1 and taxi is at one of the stations.
    if Destination_Location == [-1,-1] and destination_look==1 and (taxi_r, taxi_c) in stations:
        Destination_Location = [taxi_r, taxi_c]
        # print("Found the Destination station")
        # print(f'statios: {stations}')

    # Set the new state
    # ---------------------------
    # 1. Compute passenger differences.
    if Passenger_Location == [-1,-1]:
        pass_row_diff = 11
        pass_col_diff = 11
    else:
        pass_row_diff = taxi_r - Passenger_Location[0]
        pass_col_diff = taxi_c - Passenger_Location[1]
    
    # 2. Compute destination differences.
    if Destination_Location == [-1,-1]:
        dest_row_diff = 11
        dest_col_diff = 11
    else:
        dest_row_diff = taxi_r - Destination_Location[0]
        dest_col_diff = taxi_c - Destination_Location[1]
    
    # 3. Define picked flag.
    picked = int(Passenger_on_Taxi)

    # 4. Better going station
    # Taxi should go to the passenger station if the passenger is not picked up.
    if Passenger_Location != [-1,-1] and not Passenger_on_Taxi:
        going_row_diff = taxi_r - Passenger_Location[0]
        going_col_diff = taxi_c - Passenger_Location[1]
    elif Destination_Location != [-1,-1] and Passenger_on_Taxi:
        going_row_diff = taxi_r - Destination_Location[0]
        going_col_diff = taxi_c - Destination_Location[1]
    # Goes to the closedt station that has not been visited yet.
    elif Been_to_Stations.count(False) > 0:
        indices = [i for i, x in enumerate(Been_to_Stations) if not x]
        distances = [abs(taxi_r - stations[i][0]) + abs(taxi_c - stations[i][1]) for i in indices]
        min_index = indices[distances.index(min(distances))]
        going_row_diff = taxi_r - stations[min_index][0]
        going_col_diff = taxi_c - stations[min_index][1]
    else:
        raise ValueError("No station to go to.")
    
    # If taxi is at one of the stations been to that station.
    if (taxi_r, taxi_c) in stations:
        Been_to_Stations[stations.index((taxi_r, taxi_c))] = True
        # print("Been to station: ", (taxi_r, taxi_c))
        
    
    # Check for correct dropoff: if taxi is at Destination_Location, passenger is on taxi, and last action was dropoff.
    if Destination_Location == [taxi_r, taxi_c] and Passenger_on_Taxi and last_action == 5:
        Done_or_not = True
        # print("Passenger dropped off correctly. Resetting globals.")


    # If Fuel exceeds MAX_FUEL, mark done.
    Fuel += 1
    if Fuel > MAX_FUEL:
        Done_or_not = True
        # print("Fuel exceeded MAX_FUEL. Resetting globals.")

    new_state = (going_row_diff, going_col_diff, picked,
                 obs_n, obs_s, obs_e, obs_w)
    other_state = (pass_row_diff, pass_col_diff, dest_row_diff, dest_col_diff)
    return new_state, other_state

def reset_global_storage():
    """Reset the global storage variables for a new episode."""
    global Destination_Location, Passenger_Location, Passenger_on_Taxi, Been_to_Stations, last_action, Fuel, Done_or_not
    Destination_Location = [-1, -1]
    Passenger_Location = [-1, -1]
    Passenger_on_Taxi = False
    Been_to_Stations = [False, False, False, False]
    last_action = -1
    Fuel = 0
    Done_or_not = False
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
