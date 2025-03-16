import os
import random
import pickle
import numpy as np

# ---------------------------
# Dynamic Taxi Environment with Random Grid Size
# ---------------------------
class DynamicTaxiEnv():
    def __init__(self, grid_size_min=5, grid_size_max=15, fuel_limit=100, obstacle_prob=0.2):
        """
        grid_size_min: minimum grid size (n)
        grid_size_max: maximum grid size (n)
        fuel_limit: maximum steps (fuel) available (can be kept constant or computed as function of grid size)
        obstacle_prob: probability for any non‐station cell to be an obstacle.
        """
        self.grid_size_min = grid_size_min
        self.grid_size_max = grid_size_max
        self.fuel_limit = fuel_limit
        self.obstacle_prob = obstacle_prob
        self.passenger_picked_up = False
        self.reset()

    def select_stations(self, all_positions, num=4, min_dist=3):
        """
        Select num positions from all_positions such that every pair is at least min_dist apart (Manhattan distance).
        If no valid sample is found after many attempts, returns a random sample.
        """
        attempts = 0
        while attempts < 100000:
            candidate = random.sample(all_positions, num)
            valid = True
            for i in range(num):
                for j in range(i + 1, num):
                    if abs(candidate[i][0] - candidate[j][0]) + abs(candidate[i][1] - candidate[j][1]) < min_dist:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return candidate
            attempts += 1
        return random.sample(all_positions, num)

    def reset(self):
        # Choose a random grid size for this episode.
        self.grid_size = random.randint(self.grid_size_min, self.grid_size_max)
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False

        # Create a list of all grid positions.
        all_positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        # Define a minimum distance between stations.
        min_station_distance = max(3, self.grid_size // 2)
        # Randomly assign 4 station locations ensuring they are not too close.
        self.stations = self.select_stations(all_positions, num=4, min_dist=2)

        # Generate obstacles randomly in non‐station cells.
        self.obstacles = set()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) in self.stations:
                    continue
                if random.random() < self.obstacle_prob:
                    self.obstacles.add((i, j))

        # Passenger starting location is randomly chosen from stations.
        self.passenger_loc = random.choice(self.stations)
        # print('passenger location',self.passenger_loc)
        # Destination is randomly chosen from the remaining stations.
        possible_destinations = [s for s in self.stations if s != self.passenger_loc]
        self.destination = random.choice(possible_destinations)
        # print('destination',self.destination)
        # Taxi starts in a random free cell (avoid obstacles).
        free_positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)
                        if (i, j) not in self.obstacles and (i, j) not in self.stations]
        self.taxi_pos = random.choice(free_positions)
        return self.get_state(), {}

    def step(self, action):
        """Perform an action and update the environment state."""
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0
        if action == 0 :  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1
        
        
        if action in [0, 1, 2, 3]:  # Only movement actions should be checked
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -=5
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
        else:
            if action == 4:  # PICKUP
                if self.taxi_pos == self.passenger_loc:
                    self.passenger_picked_up = True
                    self.passenger_loc = self.taxi_pos  
                else:
                    reward = -10  
            elif action == 5:  # DROPOFF
                if self.passenger_picked_up:
                    if self.taxi_pos == self.destination:
                        reward += 50
                        return self.get_state(), reward -0.1, True, {}
                    else:
                        reward -=10
                    self.passenger_picked_up = False
                    # self.passenger_loc = self.taxi_pos
                else:
                    reward -=10
                    
        reward -= 0.1  

        self.current_fuel -= 1
        if self.current_fuel <= 0:
            return self.get_state(), reward -10, True, {}

        

        return self.get_state(), reward, False, {}

    def get_state(self):
        """Return the current environment state."""
        taxi_row, taxi_col = self.taxi_pos
        passenger_row, passenger_col = self.passenger_loc
        destination_row, destination_col = self.destination
        
        obstacle_north = int(taxi_row == 0 or (taxi_row-1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row+1, taxi_col) in self.obstacles)
        obstacle_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col+1) in self.obstacles)
        obstacle_west  = int(taxi_col == 0 or (taxi_row , taxi_col-1) in self.obstacles)

        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east  = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west  = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle  = int( (taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle
       
        destination_loc_north = int( (taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int( (taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east  = int( (taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west  = int( (taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle  = int( (taxi_row, taxi_col) == self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle

        # print('-----------from get_state-----------')
        # print('taxi_row',taxi_row)
        # print('taxi_col',taxi_col)
        # print('passenger_row',passenger_row)
        # print('passenger_col',passenger_col)
        # print('destination_row',destination_row)
        # print('destination_col',destination_col)
        # print('passenger_look',passenger_look)
        # print('destination_look',destination_look)
        # print(f'self.passenger_picked_up:{self.passenger_picked_up}')
        # print('-----------------------------------')
        
        state = (taxi_row, taxi_col, self.stations[0][0],self.stations[0][1] ,self.stations[1][0],self.stations[1][1],self.stations[2][0],self.stations[2][1],self.stations[3][0],self.stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
        return state

    def render(self):
        """
        Render a text representation of the grid.
        Stations are shown as letters R, G, Y, B (in order of self.stations).
        Obstacles are shown as X.
        Passenger (if not picked) is shown as P.
        Taxi is shown as T.
        Destination is shown as D.
        """
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        # Place obstacles.
        for (r, c) in self.obstacles:
            grid[r][c] = 'X'
        # Place stations.
        station_letters = ['R', 'G', 'Y', 'B']
        for i, s in enumerate(self.stations):
            r, c = s
            grid[r][c] = station_letters[i]
        # Mark destination (override station marker with D).
        dr, dc = self.destination
        grid[dr][dc] = 'D'
        # Place passenger if not picked up.
        if not self.passenger_picked_up:
            pr, pc = self.passenger_loc
            grid[pr][pc] = 'P'
        # Place taxi.
        tr, tc = self.taxi_pos
        grid[tr][tc] = 'T'

        out = ""
        for row in grid:
            out += " ".join(row) + "\n"
        return out
