import random
# globally storage
# ---------------------------
Destination_Location = [-1, -1]
Passenger_Location = [-1, -1]
Passenger_on_Taxi = False
Been_to_Stations = [False, False, False, False]
last_action = -1
Done_or_not = False
Fuel = 0

MAX_FUEL = 100

# ---------------------------
# New State Conversion Function for Training
# ---------------------------
def get_state_self_defined(obs):
    """
    Convert the original environment state (obs) into a new state representation for training.
    
    Original obs is a 16-tuple:
      (taxi_row, taxi_col,
       station0_row, station0_col, station1_row, station1_col,
       station2_row, stat   ion2_col, station3_row, station3_col,
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
    # Taxi should go to the passenger station if the passenger is not picked up and the passenegr location is known.
    if Passenger_Location != [-1,-1] and not Passenger_on_Taxi:
        going_row_diff = taxi_r - Passenger_Location[0]
        going_col_diff = taxi_c - Passenger_Location[1]
        print(f"**Going to passenger station [{Passenger_Location[0]},{Passenger_Location[1]}]")
        print(f'going_row_diff: {going_row_diff}, going_col_diff: {going_col_diff}')
    # Taxi should go to the destination station if the passenger is picked up and the destination location is known.
    elif Destination_Location != [-1,-1] and Passenger_on_Taxi:
        going_row_diff = taxi_r - Destination_Location[0]
        going_col_diff = taxi_c - Destination_Location[1]
        print(f"**Going to destination station [{Destination_Location[0]},{Destination_Location[1]}]")
        print(f'going_row_diff: {going_row_diff}, going_col_diff: {going_col_diff}')
    # Goes to the closedt station that has not been visited yet if the destination or passenegr location hsan't been found.
    elif Been_to_Stations.count(False) > 0 or (Destination_Location == [-1,-1] and Passenger_on_Taxi):
        indices = [i for i, x in enumerate(Been_to_Stations) if not x]
        distances = [abs(taxi_r - stations[i][0]) + abs(taxi_c - stations[i][1]) for i in indices]
        
        min_index = indices[distances.index(min(distances))]
        going_row_diff = taxi_r - stations[min_index][0]
        going_col_diff = taxi_c - stations[min_index][1]
        print(f"**Going to station [{stations[min_index][0]},{stations[min_index][1]}]")
        print(f'going_row_diff: {going_row_diff}, going_col_diff: {going_col_diff}')

    else:
        print("No station to go to.")
        # randomly pick one station to go to
        station = random.choice(stations)
        going_row_diff = taxi_r - station[0]
        going_col_diff = taxi_c - station[1]
    
    # If taxi is at one of the stations been to that station.
    if (taxi_r, taxi_c) in stations:
        Been_to_Stations[stations.index((taxi_r, taxi_c))] = True
        print("****Been to station: ", (taxi_r, taxi_c))
    
    # Check for correct dropoff: if taxi is at Destination_Location, passenger is on taxi, and last action was dropoff.
    if Destination_Location == [taxi_r, taxi_c] and Passenger_on_Taxi and last_action == 5:
        Done_or_not = True
        # print("Passenger dropped off correctly. Resetting globals.")

    if Destination_Location!=[-1,-1]:
        seen_destination = 1
    else:
        seen_destination = 0
    
    if Passenger_Location!=[-1,-1]:
        seen_passenger = 1
    else:
        seen_passenger = 0
    
    # If Fuel exceeds MAX_FUEL, mark done.
    Fuel += 1
    if Fuel > MAX_FUEL:
        Done_or_not = True
        # print("Fuel exceeded MAX_FUEL. Resetting globals.")

    new_state = (going_row_diff, going_col_diff, picked,
                 obs_n, obs_s, obs_e, obs_w, seen_destination, seen_passenger)
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



import random
import numpy as np

class StateRecorder:
    def __init__(self, max_fuel):
        """
        Initialize the state recorder with a maximum fuel value.
        """
        self.max_fuel = max_fuel
        self.reset()

    def reset(self):
        """Reset all stored variables for a new episode."""
        self.destination_location = [-1, -1]  # Destination station location.
        self.passenger_location = [-1, -1]    # Passenger station location.
        self.passenger_on_taxi = False         # Whether the passenger is on board.
        self.been_to_stations = [False, False, False, False]  # Visited flags for the 4 stations.
        self.last_action = -1                  # Last action taken (-1 means no action yet).
        self.fuel = 0                          # Step counter used as fuel.
        self.done = False                      # Flag for terminal condition.

    def update(self, obs, action):
        """
        Update the stored variables using the current observation (obs) and the action taken.
        obs is a 16-tuple:
          (taxi_row, taxi_col,
           station0_row, station0_col, station1_row, station1_col,
           station2_row, station2_col, station3_row, station3_col,
           obstacle_north, obstacle_south, obstacle_east, obstacle_west,
           passenger_look, destination_look)
        """
        taxi_r, taxi_c = obs[0], obs[1]
        stations = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]
        passenger_look = obs[14]
        destination_look = obs[15]

        # Increment fuel (i.e. step counter)
        self.fuel += 1

        # Update last action.
        self.last_action = action

        # If taxi is at a station, mark that station as visited.
        for i, s in enumerate(stations):
            if (taxi_r, taxi_c) == s:
                self.been_to_stations[i] = True
                # If passenger location is not recorded and passenger is seen, record it.
                if passenger_look == 1 and self.passenger_location == [-1, -1]:
                    self.passenger_location = [taxi_r, taxi_c]
                # Similarly, if destination location is not recorded and destination is seen.
                if destination_look == 1 and self.destination_location == [-1, -1]:
                    self.destination_location = [taxi_r, taxi_c]

        # Update passenger status.
        # If taxi is at the recorded passenger location and action is pickup (4), mark passenger on taxi.
        if (self.passenger_location != [-1, -1] and 
            (taxi_r, taxi_c) == tuple(self.passenger_location) and 
            action == 4):
            self.passenger_on_taxi = True

        # If passenger is on taxi, update passenger location to current taxi position.
        if self.passenger_on_taxi:
            self.passenger_location = [taxi_r, taxi_c]
            # If dropoff (action 5) at the destination, mark terminal condition.
            if (action == 5 and 
                self.destination_location != [-1, -1] and 
                (taxi_r, taxi_c) == tuple(self.destination_location)):
                self.done = True
                # print("Passenger dropped off correctly. Resetting global storage.")
                self.reset()
                return

            # If dropoff occurs but taxi is not at destination, passenger is dropped off incorrectly.
            if action == 5 and (self.destination_location == [-1, -1] or (taxi_r, taxi_c) != tuple(self.destination_location)):
                self.passenger_on_taxi = False
                # print("Passenger dropped off at wrong location.")

        # Check fuel condition.
        if self.fuel > self.max_fuel:
            self.done = True
            # print("Fuel exceeded MAX_FUEL. Resetting global storage.")
            self.reset()

    def get_visited_code(self):
        """
        Encode the been_to_stations list as a 4-bit integer (0 to 15) and normalize.
        """
        code = self.been_to_stations[0] * 8 + self.been_to_stations[1] * 4 + self.been_to_stations[2] * 2 + self.been_to_stations[3]
        return code / 15.0

    def get_state(self, obs):
        """
        Produce a new state representation using the current observation (obs) and the stored history.
        
        We'll define the new state as a 10-tuple:
          (going_row_diff, going_col_diff, picked, last_action_norm, visited_code_norm, fuel_norm, obs_n, obs_s, obs_e, obs_w)
          
        where:
          - The target station is chosen as follows:
              • If passenger not on taxi and passenger_location is known, use passenger_location.
              • If passenger is on taxi and destination_location is known, use destination_location.
              • Otherwise, choose the nearest unvisited station.
          - going_row_diff and going_col_diff are the differences between taxi position and target station.
          - picked is 1 if passenger_on_taxi is True, else 0.
          - last_action_norm is last_action normalized by 5 (if last_action is -1, use 0).
          - visited_code_norm is obtained from get_visited_code().
          - fuel_norm is fuel divided by max_fuel.
          - The obstacle indicators (obs_n, obs_s, obs_e, obs_w) come from the original obs.
        """
        taxi_r, taxi_c = obs[0], obs[1]
        stations = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]

        # Determine target station.
        if not self.passenger_on_taxi:
            if self.passenger_location != [-1, -1]:
                target = self.passenger_location
            else:
                # filter out the stations that have been visited
                unvisited_stations = [s for i, s in enumerate(stations) if not self.been_to_stations[i]]
                if len(unvisited_stations) > 0:
                    distances = [abs(taxi_r - s[0]) + abs(taxi_c - s[1]) for s in unvisited_stations]
                    target = unvisited_stations[int(np.argmin(distances))]
                else:
                    print("No station to go to.")
                    # random pick one station to go to
                    target = random.choice(stations)
        else:
            if self.destination_location != [-1, -1]:
                target = self.destination_location
            else:
                # filter out the stations that have been visited
                unvisited_stations = [s for i, s in enumerate(stations) if not self.been_to_stations[i]]
                if len(unvisited_stations) > 0:
                    distances = [abs(taxi_r - s[0]) + abs(taxi_c - s[1]) for s in unvisited_stations]
                    target = unvisited_stations[int(np.argmin(distances))]
                else:
                    print("No station to go to.")
                    # random pick one station to go to
                    target = random.choice(stations)

        going_row_diff = taxi_r - target[0]
        going_col_diff = taxi_c - target[1]
        picked = 1 if self.passenger_on_taxi else 0
        last_action_norm = 0 if self.last_action == -1 else self.last_action / 5.0
        visited_code_norm = self.get_visited_code()
        fuel_norm = self.fuel / self.max_fuel

        obs_n = obs[10]
        obs_s = obs[11]
        obs_e = obs[12]
        obs_w = obs[13]

        state = (going_row_diff, going_col_diff, picked, last_action_norm, visited_code_norm, fuel_norm, obs_n, obs_s, obs_e, obs_w)
        # state = (going_row_diff, going_col_diff, picked, obs_n, obs_s, obs_e, obs_w)
        # other state is a 4-tuple: (pass_row_diff, pass_col_diff, dest_row_diff, dest_col_diff)
        if self.passenger_location != [-1, -1]:
            pass_row_diff = taxi_r - self.passenger_location[0]
            pass_col_diff = taxi_c - self.passenger_location[1]
        else:
            pass_row_diff, pass_col_diff = 11, 11
        
        if self.destination_location != [-1, -1]:
            dest_row_diff = taxi_r - self.destination_location[0]
            dest_col_diff = taxi_c - self.destination_location[1]
        else:
            dest_row_diff, dest_col_diff = 11, 11
        
        other_state = (pass_row_diff, pass_col_diff, dest_row_diff, dest_col_diff)
        return state, other_state
    
    def normalize_state(self, state):
        """
        Normalize the state values for neural network training.
        """
        going_row_diff, going_col_diff, picked, last_action_norm, visited_code_norm, fuel_norm, obs_n, obs_s, obs_e, obs_w = state
        return (going_row_diff / 10.0, going_col_diff / 10.0, picked, last_action_norm, visited_code_norm, fuel_norm, obs_n, obs_s, obs_e, obs_w)
        
# Create a global instance of the recorder.
global_state_recorder = StateRecorder(MAX_FUEL)


import random

class StateRecorder_LargeState:
    def __init__(self, max_fuel):
        """
        Initialize the state recorder with a maximum fuel value.
        """
        self.max_fuel = max_fuel
        self.reset()

    def reset(self):
        """
        Reset all stored variables for a new episode.
        """
        self.destination_location = [-1, -1]  # Destination station location.
        self.passenger_location = [-1, -1]    # Passenger station location.
        self.passenger_on_taxi = False        # Whether the passenger is on board.
        self.been_to_stations = [False, False, False, False]  # Visited flags for the 4 stations.
        self.last_action = -1                 # Last action taken (-1 means no action yet).
        self.fuel = 0                         # Step counter used as fuel.
        self.done = False                     # Terminal flag.

    def update(self, obs, action):
        """
        Update stored variables using the current observation (obs) and the action taken.
        The observation is a 16-tuple:
          (taxi_row, taxi_col,
           station0_row, station0_col, station1_row, station1_col,
           station2_row, station2_col, station3_row, station3_col,
           obstacle_north, obstacle_south, obstacle_east, obstacle_west,
           passenger_look, destination_look)
        """
        taxi_r, taxi_c = obs[0], obs[1]
        stations = [(obs[2], obs[3]), (obs[4], obs[5]),
                    (obs[6], obs[7]), (obs[8], obs[9])]
        passenger_look = obs[14]
        destination_look = obs[15]

        # Increment fuel counter.
        self.fuel += 1

        # Update last action.
        self.last_action = action

        # Update visited stations and record passenger/destination positions if observed.
        for i, station in enumerate(stations):
            if (taxi_r, taxi_c) == station:
                self.been_to_stations[i] = True
                if passenger_look == 1 and self.passenger_location == [-1, -1]:
                    self.passenger_location = [taxi_r, taxi_c]
                if destination_look == 1 and self.destination_location == [-1, -1]:
                    self.destination_location = [taxi_r, taxi_c]

        # Update passenger status.
        if (self.passenger_location != [-1, -1] and 
            (taxi_r, taxi_c) == tuple(self.passenger_location) and
            action == 4):  # PICKUP action
            self.passenger_on_taxi = True

        if self.passenger_on_taxi:
            # When passenger is onboard, update its location to taxi's current position.
            self.passenger_location = [taxi_r, taxi_c]
            if action == 5:  # DROPOFF action
                if (self.destination_location != [-1, -1] and 
                    (taxi_r, taxi_c) == tuple(self.destination_location)):
                    self.done = True
                    self.reset()
                    return
                else:
                    # Incorrect dropoff resets the passenger flag.
                    self.passenger_on_taxi = False

        # Check fuel condition.
        if self.fuel > self.max_fuel:
            self.done = True
            self.reset()

    def get_visited_code(self):
        """
        Encode the been_to_stations list as a 4-bit integer (0 to 15) and normalize it.
        """
        code = (self.been_to_stations[0] * 8 +
                self.been_to_stations[1] * 4 +
                self.been_to_stations[2] * 2 +
                self.been_to_stations[3])
        return code / 15.0

    def get_state(self, obs):
        """
        Generate a state representation based solely on the observation and stored history,
        without any heuristic guidance for a target location.

        The state vector contains:
          - Taxi's normalized absolute position (taxi_row/10, taxi_col/10).
          - Passenger information: a binary flag indicating if the passenger location is known,
            and the normalized relative differences (taxi - passenger). If unknown, use a sentinel value of 2.0.
          - Destination information: a binary flag and normalized differences (taxi - destination),
            with sentinel values if unknown.
          - Normalized visited stations code.
          - Normalized last action (action/5.0).
          - Normalized fuel usage.
          - Obstacle indicators: obs_n, obs_s, obs_e, obs_w.
          - Raw passenger_look and destination_look flags.
        """
        taxi_r, taxi_c = obs[0], obs[1]

        # Taxi's normalized absolute position.
        norm_taxi_r = taxi_r / 10.0
        norm_taxi_c = taxi_c / 10.0

        # Passenger relative information.
        if self.passenger_location != [-1, -1]:
            passenger_known = 1.0
            pass_diff_r = (taxi_r - self.passenger_location[0]) / 10.0
            pass_diff_c = (taxi_c - self.passenger_location[1]) / 10.0
        else:
            passenger_known = 0.0
            pass_diff_r = 2.0  # Sentinel value outside the normal range [-1,1].
            pass_diff_c = 2.0

        # Destination relative information.
        if self.destination_location != [-1, -1]:
            destination_known = 1.0
            dest_diff_r = (taxi_r - self.destination_location[0]) / 10.0
            dest_diff_c = (taxi_c - self.destination_location[1]) / 10.0
        else:
            destination_known = 0.0
            dest_diff_r = 2.0  # Sentinel value.
            dest_diff_c = 2.0

        # Visited stations encoded as a normalized code.
        visited_code_norm = self.get_visited_code()

        # Last action normalized (if no action, use 0).
        last_action_norm = 0.0 if self.last_action == -1 else self.last_action / 5.0

        # Normalized fuel usage.
        fuel_norm = self.fuel / self.max_fuel

        # Passenger on taxi flag.
        passenger_on = 1.0 if self.passenger_on_taxi else 0.0

        # Obstacle indicators (indices 10-13 in the observation).
        obs_n = obs[10]
        obs_s = obs[11]
        obs_e = obs[12]
        obs_w = obs[13]

        # Raw look flags for passenger and destination.
        passenger_look = obs[14]
        destination_look = obs[15]

        # Assemble the complete state vector.
        state = (
            norm_taxi_r, norm_taxi_c,
            passenger_known, pass_diff_r, pass_diff_c,
            destination_known, dest_diff_r, dest_diff_c,
            passenger_on,
            visited_code_norm,
            last_action_norm,
            obs_n, obs_s, obs_e, obs_w,
            passenger_look, destination_look
        )
        return state

global_state_recorder_large = StateRecorder_LargeState(MAX_FUEL)

class StateRecorder_Enhanced:
    def __init__(self, max_fuel):
        """
        Initialize the state recorder with a maximum fuel value.
        """
        self.max_fuel = max_fuel
        self.reset()

    def reset(self):
        """
        Reset all stored variables for a new episode.
        """
        self.taxi_pos = None  # Will be updated on first call.
        self.station_positions = None  # List of 4 station positions.
        self.passenger_station_index = -1  # Index (0-3) if passenger station is known.
        self.destination_station_index = -1  # Index (0-3) if destination is known.
        self.passenger_on_taxi = False  # Whether the passenger is on board.
        self.last_action = -1  # Last action taken (-1 means none).
        self.fuel = 0  # Step counter (used as fuel).
        self.done = False  # Terminal flag.

    def update(self, obs, action):
        """
        Update stored variables using the current observation (obs) and the action taken.
        The observation is a 16-tuple:
          (taxi_row, taxi_col,
           station0_row, station0_col, station1_row, station1_col,
           station2_row, station2_col, station3_row, station3_col,
           obstacle_north, obstacle_south, obstacle_east, obstacle_west,
           passenger_look, destination_look)
        """
        taxi_r, taxi_c = obs[0], obs[1]
        self.taxi_pos = (taxi_r, taxi_c)

        # Record the station positions (assumed fixed during the episode).
        if self.station_positions is None:
            self.station_positions = [(obs[2], obs[3]), (obs[4], obs[5]),
                                      (obs[6], obs[7]), (obs[8], obs[9])]

        passenger_look = obs[14]
        destination_look = obs[15]
        self.last_action = action
        self.fuel += 1

        # When the taxi is at a station, record which station is seen as passenger/destination.
        for i, pos in enumerate(self.station_positions):
            if (taxi_r, taxi_c) == pos:
                if passenger_look == 1 and self.passenger_station_index == -1:
                    self.passenger_station_index = i
                if destination_look == 1 and self.destination_station_index == -1:
                    self.destination_station_index = i

        # Update passenger status: if a pickup (action 4) occurs at the passenger station, mark as onboard.
        if action == 4 and self.passenger_station_index != -1:
            if (taxi_r, taxi_c) == self.station_positions[self.passenger_station_index]:
                self.passenger_on_taxi = True

        # If the taxi attempts a dropoff (action 5) at the destination station, finish the episode.
        if self.passenger_on_taxi and action == 5:
            if self.destination_station_index != -1 and (taxi_r, taxi_c) == self.station_positions[self.destination_station_index]:
                self.done = True
                # Optionally, you can reset internal history here or leave it for the training loop.
            else:
                # Incorrect dropoff resets the onboard flag.
                self.passenger_on_taxi = False

        if self.fuel > self.max_fuel:
            self.done = True
            self.reset()

    def get_state(self, obs):
        """
        Generate a state representation using the current observation and stored history.
        
        New state vector components (all values normalized assuming max grid size of 10):
          1. Taxi absolute position (2 dims): taxi_row/10, taxi_col/10.
          2. Station positions (8 dims): For each station, station_row/10 and station_col/10.
          3. Passenger indicator (4 dims): One-hot vector for which station is the passenger station, if known.
          4. Destination indicator (4 dims): One-hot vector for which station is the destination station, if known.
          5. Passenger on taxi flag (1 dim): 1 if passenger onboard, 0 otherwise.
          6. Fuel level (1 dim): fuel / max_fuel.
          7. Last action (1 dim): last_action normalized by 5.
          8. Obstacle indicators (4 dims): obs_n, obs_s, obs_e, obs_w.
          9. Raw look flags (2 dims): passenger_look and destination_look.
        
        Total dimensions: 2+8+4+4+1+1+1+4+2 = 27.
        """
        taxi_r, taxi_c = obs[0], obs[1]
        norm_taxi_r = taxi_r / 10.0
        norm_taxi_c = taxi_c / 10.0

        # Station positions.
        self.station_positions = [(obs[2], obs[3]), (obs[4], obs[5]),
                                      (obs[6], obs[7]), (obs[8], obs[9])]
        station_features = []
        for (r, c) in self.station_positions:
            station_features.extend([r / 10.0, c / 10.0])

        # Passenger one-hot indicator.
        passenger_one_hot = [0.0] * 4
        if self.passenger_station_index != -1:
            passenger_one_hot[self.passenger_station_index] = 1.0

        # Destination one-hot indicator.
        destination_one_hot = [0.0] * 4
        if self.destination_station_index != -1:
            destination_one_hot[self.destination_station_index] = 1.0

        passenger_on = 1.0 if self.passenger_on_taxi else 0.0
        fuel_norm = self.fuel / self.max_fuel
        last_action_norm = 0.0 if self.last_action == -1 else self.last_action / 5.0

        # Obstacle indicators (indices 10-13).
        obstacles = list(obs[10:14])
        # Raw look flags (indices 14-15).
        raw_flags = list(obs[14:16])

        # Assemble state vector.
        state_vector = (
            [norm_taxi_r, norm_taxi_c] +
            station_features +
            passenger_one_hot +
            destination_one_hot +
            [passenger_on, last_action_norm] +
            obstacles +
            raw_flags
        )
        return state_vector

global_state_recorder_enhanced = StateRecorder_Enhanced(MAX_FUEL)

class StateRecorder_Differences:
    def __init__(self, max_fuel):
        """
        Initialize the state recorder with a maximum fuel value.
        """
        self.max_fuel = max_fuel
        self.reset()
    
    def reset(self):
        """
        Reset all stored variables for a new episode.
        """
        self.passenger_location = None       # (row, col) when known
        self.destination_location = None     # (row, col) when known
        self.passenger_on_taxi = False         # Whether the passenger is on board
        self.last_action = -1                  # Last action taken (-1 means none)
        self.fuel = 0                          # Step counter (used as fuel)
        self.done = False                      # Terminal flag
    
    def update(self, obs, action):
        """
        Update stored variables using the current observation (obs) and the action taken.
        The observation is a 16-tuple:
          (taxi_row, taxi_col,
           station0_row, station0_col, station1_row, station1_col,
           station2_row, station2_col, station3_row, station3_col,
           obstacle_north, obstacle_south, obstacle_east, obstacle_west,
           passenger_look, destination_look)
        """
        taxi_r, taxi_c = obs[0], obs[1]
        passenger_look = obs[14]
        destination_look = obs[15]
        
        self.last_action = action
        self.fuel += 1
        
        # When the taxi is at a station and the corresponding look flag is on,
        # record the passenger or destination location if not already known.
        # (Assuming that when a look flag is 1, the taxi is at the relevant station.)
        if passenger_look == 1 and self.passenger_location is None:
            self.passenger_location = (taxi_r, taxi_c)
        if destination_look == 1 and self.destination_location is None:
            self.destination_location = (taxi_r, taxi_c)
        
        # If a pickup action (4) is taken when the taxi is at the recorded passenger location,
        # mark that the passenger is onboard.
        if action == 4 and self.passenger_location is not None:
            if (taxi_r, taxi_c) == self.passenger_location:
                self.passenger_on_taxi = True
        
        # If a dropoff action (5) is taken when the taxi is at the recorded destination location,
        # mark the episode as done.
        if self.passenger_on_taxi and action == 5:
            if self.destination_location is not None and (taxi_r, taxi_c) == self.destination_location:
                self.done = True
                self.reset()  # Optionally, you might not want to reset immediately here.
            else:
                self.passenger_on_taxi = False
        
        if self.fuel > self.max_fuel:
            self.done = True
            self.reset()
    
    def get_state(self, obs):
        """
        Generate a state representation based solely on the observation and stored history.
        
        The state vector is constructed as differences:
          1. Passenger differences: 
             [ (taxi_row - passenger_row)/10, (taxi_col - passenger_col)/10, passenger_known ]
          2. Destination differences:
             [ (taxi_row - destination_row)/10, (taxi_col - destination_col)/10, destination_known ]
          3. Passenger-to-Destination differences:
             [ (passenger_row - destination_row)/10, (passenger_col - destination_col)/10, both_known ]
          4. Passenger on flag (1 dim)
          5. Last action normalized (1 dim, divided by 5)
          6. Fuel usage normalized (1 dim)
          7. Obstacle indicators (4 dims) from obs[10:14]
        
        Total dimensions: 3 + 3 + 3 + 1 + 1 + 1 + 4 = 16.
        """
        taxi_r, taxi_c = obs[0], obs[1]
        norm_factor = 10.0
        
        # Passenger difference.
        if self.passenger_location is not None:
            pass_diff_r = (taxi_r - self.passenger_location[0]) / norm_factor
            pass_diff_c = (taxi_c - self.passenger_location[1]) / norm_factor
            passenger_known = 1.0
        else:
            pass_diff_r, pass_diff_c = 2.0, 2.0  # Sentinel values outside the typical range [-1,1]
            passenger_known = 0.0
        
        # Destination difference.
        if self.destination_location is not None:
            dest_diff_r = (taxi_r - self.destination_location[0]) / norm_factor
            dest_diff_c = (taxi_c - self.destination_location[1]) / norm_factor
            destination_known = 1.0
        else:
            dest_diff_r, dest_diff_c = 2.0, 2.0
            destination_known = 0.0
        
        # Passenger-to-Destination difference.
        if self.passenger_location is not None and self.destination_location is not None:
            pd_diff_r = (self.passenger_location[0] - self.destination_location[0]) / norm_factor
            pd_diff_c = (self.passenger_location[1] - self.destination_location[1]) / norm_factor
            both_known = 1.0
        else:
            pd_diff_r, pd_diff_c = 2.0, 2.0
            both_known = 0.0
        
        # Passenger on flag.
        passenger_on_flag = 1.0 if self.passenger_on_taxi else 0.0
        
        # Last action normalized.
        last_action_norm = 0.0 if self.last_action == -1 else self.last_action / 5.0
        
        # Fuel usage normalized.
        fuel_norm = self.fuel / self.max_fuel
        
        # Obstacle indicators: obs[10:14]
        obstacles = list(obs[10:14])
        
        # Assemble state vector.
        state_vector = [
            pass_diff_r, pass_diff_c, passenger_known,
            dest_diff_r, dest_diff_c, destination_known,
            pd_diff_r, pd_diff_c, both_known,
            passenger_on_flag,
            last_action_norm
        ] + obstacles  # obstacles adds 4 dimensions
        
        return state_vector

global_state_recorder_Differnces = StateRecorder_Differences(MAX_FUEL)