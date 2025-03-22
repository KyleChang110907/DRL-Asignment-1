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

# SARSA2 
from self_defined_state import global_state_recorder  # assume you save the above class in state_recorder.py

def get_action_sarsa(obs):
    """
    Given an observation, convert it to the new state using the global StateRecorder,
    then load the Q-table from file and return the action with the highest Q-value.
    If the new state is not found in the Q-table, return a random action.
    """
    # Get the new state representation.
    new_state, new_other_state = global_state_recorder.get_state(obs)
    
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

# A3C
# Load the stored A3C network.
# This global network is loaded only once.
from training.A3C_LargeState import A3CNet
import torch
import torch.nn.functional as F
from self_defined_state import global_state_recorder_large

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_net = A3CNet(input_dim=18, hidden_dim=128, num_actions=6).to(device)
global_net.load_state_dict(torch.load("./results_dynamic/a3c_policy_net.pt", map_location=device))
global_net.eval()

def get_action(obs):
    """
    Given an observation (obs), convert it into the new state using the global StateRecorder,
    then use the stored A3C network to compute the policy distribution, sample an action,
    update the recorder, and return the action.
    """
    # Convert the raw observation into your new state representation.
    # global_state_recorder.get_state returns a 10-tuple.
    new_state= global_state_recorder_large.get_state(obs)
    
    # Convert state to tensor.
    state_tensor = torch.tensor(new_state, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Pass through the global network.
    with torch.no_grad():
        policy_logits, _ = global_net(state_tensor)
    # Compute action probabilities and sample.
    probs = F.softmax(policy_logits, dim=1)
    m = torch.distributions.Categorical(probs)
    action = int(m.sample().item())
    
    # Update the state recorder with the current observation and chosen action.
    global_state_recorder_large.update(obs, action)
    
    return action
# def get_action(obs):
#     return random.choice(range(6))


# DQN Large State
import torch
import torch.nn.functional as F
from self_defined_state import global_state_recorder_large  # global instance of your state recorder
from training.DQN_LargeState import QNet  # assuming you have defined QNet as your DQN network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the trained Q-network.
global_net = QNet(input_dim=17, hidden_dim=128, num_actions=6).to(device)
global_net.load_state_dict(torch.load("./results_dynamic/dqn_policy_net.pt", map_location=device))
global_net.eval()

def get_action(obs):
    """
    Given an observation, convert it into your state representation via the global state recorder,
    then use the loaded Q-network to compute Q-values and select the best action (greedy).
    The state recorder is updated with the chosen action.
    """
    # Get the state representation (an 18-dimensional vector) from the state recorder.
    state = global_state_recorder_large.get_state(obs)
    
    # Convert the state to a tensor.
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Pass through the Q-network to obtain Q-values.
    with torch.no_grad():
        q_values = global_net(state_tensor)
    
    # Choose the action with the highest Q-value.
    action = int(torch.argmax(q_values, dim=1).item())
    
    # Update the state recorder with the observation and chosen action.
    global_state_recorder_large.update(obs, action)
    
    return action
