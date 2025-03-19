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
        state = (going_row_diff, going_col_diff, picked, obs_n, obs_s, obs_e, obs_w)
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
        
# Create a global instance of the recorder.
global_state_recorder = StateRecorder(MAX_FUEL)
