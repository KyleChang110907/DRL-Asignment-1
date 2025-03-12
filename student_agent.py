# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

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
#     # You can submit this random agent to evaluate the performance of a purely random strategy.

def get_action(obs):
    """
    Given an observation, load the Q-table from file and return the action
    with the highest Q-value. If the observation is not found, fallback to a random action.
    """
    try:
        with open("./results/q_table.pkl", "rb") as f:
            q_table = pickle.load(f)
    except FileNotFoundError:
        # If the Q-table hasn't been saved yet, use a random action.
        return random.choice(range(6))
    
    # Check if the observation exists in the Q-table
    if obs in q_table:
        return int(np.argmax(q_table[obs]))
    else:
        return random.choice(range(6))
