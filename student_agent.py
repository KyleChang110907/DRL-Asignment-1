# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import torch
# from self_defined_state import get_state_self_defined  
# globally storage
# ---------------------------
Destination_Location = [-1, -1]
Passenger_Location = [-1, -1]
Passenger_on_Taxi = False
Been_to_Stations = [False, False, False, False]
last_action = -1
Done_or_not = False
Fuel = 0

MAX_FUEL = 5000

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


# def get_action(obs):
    
#     # TODO: Train your own agent
#     # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
#     # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
#     #       To prevent crashes, implement a fallback strategy for missing keys. 
#     #       Otherwise, even if your agent performs well in training, it may fail during testing.
#     try:
#         with open("./results/q_table.pkl", "rb") as f:
#             q_table = pickle.load(f)
#     except FileNotFoundError:
#         return random.choice(range(6))
    
#     state_features = extract_features(obs)
#     if state_features in q_table:
#         return int(np.argmax(q_table[state_features]))
#     else:
#         return random.choice(range(6))
# #     # You can submit this random agent to evaluate the performance of a purely random strategy.

# def get_action(obs):
#     """
#     Given an observation, load the Q-table from file and return the action
#     with the highest Q-value. If the observation is not found, fallback to a random action.
#     """
#     try:
#         with open("./results/q_table.pkl", "rb") as f:
#             q_table = pickle.load(f)
#     except FileNotFoundError:
#         # If the Q-table hasn't been saved yet, use a random action.
#         return random.choice(range(6))
    
#     # Check if the observation exists in the Q-table
#     if obs in q_table:
#         return int(np.argmax(q_table[obs]))
#     else:
#         return random.choice(range(6))

# # DQN
# def get_action(obs):
#     """
#     Given an observation (state vector), load the trained DQN model from file
#     and return the action with the highest Q-value.
#     If any error occurs, falls back to a random action.
    
#     Assumes:
#       - obs is a tuple or list of length 10 (new state representation)
#       - The model is saved at "./results_dynamic/dqn_policy_net.pt"
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     try:
#         model = DQN(input_dim=10, output_dim=6).to(device)
#         model.load_state_dict(torch.load("./results_dynamic/dqn_policy_net.pt", map_location=device))
#         model.eval()
#     except Exception as e:
#         print("Error loading DQN model:", e)
#         return random.choice(range(6))
    
#     with torch.no_grad():
#         state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
#         q_vals = model(state_tensor)
#         action = int(torch.argmax(q_vals, dim=1).item())
#     return action

# SARSA
# def get_action(obs):
#     """
#     Given an observation (obs), convert it to the new state using get_state_self_defined,
#     then load the Q-table from file and return the action with the highest Q-value.
#     If the state is not found in the Q-table, return a random action.
#     """
#     # Convert the observation into the new state representation.
#     new_state, _ = get_state_self_defined(obs)
    
#     # Load the trained Q-table.
#     try:
#         with open("./results_dynamic/q_table_sarsa.pkl", "rb") as f:
#             q_table = pickle.load(f)
#     except FileNotFoundError:
#         return random.choice(range(6))
    
#     # If the new state exists in the Q-table, select the action with the highest Q-value.
#     if new_state in q_table:
#         return int(np.argmax(q_table[new_state]))
    
#     else:
#         return random.choice(range(6))

def get_action(obs):
    return random.choice(range(6))