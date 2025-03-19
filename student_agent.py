# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import torch
from self_defined_state import get_state_self_defined  
import self_defined_state

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
#         with open("./results_dynamic/q_table_sarsa3.pkl", "rb") as f:
#             q_table = pickle.load(f)
#     except FileNotFoundError:
#         action = random.choice(range(6))
        
    
#     # If the new state exists in the Q-table, select the action with the highest Q-value.
#     if new_state in q_table:
#         action = int(np.argmax(q_table[new_state]))
        
    
#     else:
#         action = random.choice(range(6))
    
#     self_defined_state.last_action = action

#     return action

from self_defined_state import global_state_recorder  # assume you save the above class in state_recorder.py

def get_action(obs):
    """
    Given an observation, convert it to the new state using the global StateRecorder,
    then load the Q-table from file and return the action with the highest Q-value.
    If the new state is not found in the Q-table, return a random action.
    """
    # Get the new state representation.
    new_state = global_state_recorder.get_state(obs)
    
    try:
        with open("./results_dynamic/q_table_sarsa4.pkl", "rb") as f:
            q_table = pickle.load(f)
    except Exception as e:
        print("Error loading Q-table:", e)
        return random.choice(range(6))
    
    if new_state in q_table:
        action = int(np.argmax(q_table[new_state]))
    else:
        action = random.choice(range(6))
        print("Action not found in Q-table. Choosing random action.")
    
    # Update the recorder with the current observation and the chosen action.
    global_state_recorder.update(obs, action)
    
    return action

# def get_action(obs):
#     return random.choice(range(6))