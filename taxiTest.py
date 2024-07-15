import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy
from colorama import Fore


class TaxiDriver:
    def __init__(self):
        self.position = [0, 0]
        self.has_passenger = False
        self.grid = []
        self.goal_finished = False
        self.current_turn = 0
        self.c_picked = -1
        self.pos_goal = False

    def restart_round(self, starting_location, grid):
        self.position = starting_location
        self.has_passenger = False
        self.grid = grid
        self.goal_finished = False
        self.current_turn = 0
        self.c_picked = -1
        self.pos_goal = False

    # If the new position the taxi moves has a customer, pick them up
    def pickup_passenger(self, new_pos):
        if 'C' in get_pos_value(self.grid, new_pos):
            self.has_passenger = True
            # add_value_to_pos(self.grid, self.position, '0T')
            self.c_picked = self.current_turn
        return self.has_passenger

    # If the new position the taxi moves has a goal and the taxi has a customer, finish
    def drop_passenger(self, new_pos):
        if 'G' in get_pos_value(self.grid, new_pos):
            if self.has_passenger:
                self.goal_finished = True
            # add_value_to_pos(self.grid, self.position, 'T')
        return self.goal_finished

    def move(self, direction):
        if self.pos_goal:
            old_pos_value = 'G'
        else:
            old_pos_value = '0'
        # old_pos_value = get_pos_value(self.grid, self.position)
        new_pos = None
        if direction == 'up':
            new_pos = [self.position[0] - 1, self.position[1]]
        elif direction == 'right':
            new_pos = [self.position[0], self.position[1] + 1]
        elif direction == 'down':
            new_pos = [self.position[0] + 1, self.position[1]]
        elif direction == 'left':
            new_pos = [self.position[0], self.position[1] - 1]
        else:
            self.choice()
        if not new_pos:
            return True
        new_pos_value = get_pos_value(self.grid, new_pos)

        if new_pos_value != '1':
            self.pos_goal = False

        if new_pos_value == 'C':
            self.pickup_passenger(new_pos)
        elif new_pos_value == 'G':
            self.drop_passenger(new_pos)
            self.pos_goal = True

        if new_pos_value != '1':
            # Update previous loc value
            add_value_to_pos(self.grid, self.position, old_pos_value)
            self.position = new_pos
            # Update current loc value
            add_value_to_pos(self.grid, self.position, 'T')

        return get_pos_value(self.grid, new_pos) != '1'

    def choice(self):
        self.current_turn += 1
        if self.current_turn >= MAX_MOVES:
            return False
        print_grid(self.grid)
        choice = input(f"Choose an action (up/right/down/left): ").lower()
        if not self.move(choice):
            return True

        print_grid(self.grid)
        return True

class TaxiAgent(TaxiDriver):

    def __init__(self, qtable={}, initial_epsilon=0.3, min_epsilon=0.01, epsilon_decay=0.99995, 
                 initial_learning_rate=1.0, min_learning_rate=0.1, learning_rate_decay=0.99995):
        super().__init__()
        self.transitions = []
        self.episode_rewards = 0
        self.is_training = True
        self.qtable = copy.deepcopy(qtable)
        self.initial_epsilon = initial_epsilon
        self.initial_learning_rate = initial_learning_rate
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.episode = 0
        self.best_reward = float('-inf')
        self.recent_rewards = []
        self.turns_without_increasing = 0

    def restart_round(self, starting_location, grid):
        super().restart_round(starting_location, grid)
        self.episode_rewards = 0
        self.transitions = []
        self.episode += 1

        # Adjust epsilon and learning rate based on performance
        if self.episode_rewards > self.best_reward:
            self.best_reward = self.episode_rewards
            # Increase epsilon and learning rate to capitalize on new better path
            self.epsilon = min(self.initial_epsilon, self.epsilon * 1.1)
            self.learning_rate = min(self.initial_learning_rate, self.learning_rate * 1.1)
            self.turns_without_increasing = 0
        else:
            self.turns_without_increasing += 1
            # Normal decay
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            self.learning_rate = max(self.min_learning_rate, self.learning_rate * self.learning_rate_decay)
        
        # If the agent has not improved in 50 episodes, set epsilon to 1.0
        # if self.turns_without_increasing >= 100000:
        #     self.turns_without_increasing = 0
        #     self.epsilon = self.initial_epsilon

        self.recent_rewards.append(self.episode_rewards)

    def store_transition(self, state, action):
        self.transitions.append((state, action))

    def calculate_discounted_rewards_and_update_q_table(self, final_reward):
        cumulative_reward = final_reward

        # Iterate through the transitions in reverse order
        for state, action in reversed(self.transitions):
            self.qtable[state][action] += self.learning_rate * cumulative_reward
            cumulative_reward = DISCOUNT_FACTOR * cumulative_reward

    def get_q_value(self, state, action):
        return self.qtable.get((state, action), 0)

    def calculate_final_reward(self, grid):
        customer_exists = any('C' in row for row in grid)
        goal_exists = any('G' in row for row in grid)

        reward = 0

        # If customer is not on the grid, add +10 to the reward
        if not customer_exists:
            reward += (MAX_MOVES - self.current_turn) * 10

        # If goal is not on the grid, add +20 to the reward
        if not goal_exists:
            reward += (MAX_MOVES - self.current_turn) * 20
        
        # If customer exists give reward based on distance (closer is better)
        if customer_exists:
            customer_pos = find_pos_of_value(grid, 'C')
            agent_pos = find_pos_of_value(grid, 'T')
            distance = abs(customer_pos[0] - agent_pos[0]) + abs(customer_pos[1] - agent_pos[1])
            # Reward is length - distance
            reward += (len(grid) - distance) * 2
        
        # If goal exists give reward based on distance (closer is better)
        if goal_exists and not customer_exists:
            goal_pos = find_pos_of_value(grid, 'G')
            agent_pos = find_pos_of_value(grid, 'T')
            distance = abs(goal_pos[0] - agent_pos[0]) + abs(goal_pos[1] - agent_pos[1])
            # Reward is (length - distance) * 2
            reward += (len(grid) - distance) * 2

        # Subtract the number of turns taken by the agent
        reward -= self.current_turn

        # If no goal was reached, add -10 to the reward
        if not customer_exists and not goal_exists:
            reward -= MAX_MOVES

        return reward

    def get_best_action(self, state):

        if state not in self.qtable:
            # If the state is not in the Q-Table, create a key with 0 values for all possible actions
            self.qtable[state] = {a: 0 for a in ACTIONS}

        if ((random.random() < self.epsilon) and self.is_training):
            # Exploration: with probability epsilon, choose a random action
            return random.choice(ACTIONS)
        else:
            best_action = max(self.qtable[state], key=self.qtable[state].get)
            return best_action

    def choice(self):
        self.current_turn += 1
        if self.current_turn >= MAX_MOVES:
            return False

        # Get the current state based on the agent's position and surroundings
        state = get_state(self.grid)
        # Use the Q-learning agent to choose the best action
        action = self.get_best_action(state)
        self.store_transition(state, action)
        self.move(action)

        return True

    def save_Q_TABLE(self, filename='qtable.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.qtable, f)
    
    def set_training(self, bool=True):
        self.is_training = bool
    
    def train_agent(self, map, num_episodes):
        self.set_training(True)
        print("Starting agent training...")
        best_reward = float('-inf')
        for episode in range(1, (num_episodes + 1)):
            # Copy map grid
            if episode % 100 == 0:
                print("Episode = ", episode)
            current_grid = copy.deepcopy(map)

            # Find taxi position
            taxi_pos = (0, 0)
            for j, row in enumerate(current_grid):
                if 'T' in row:
                    taxi_pos = (j, row.index('T'))
                    break

            self.restart_round(taxi_pos, current_grid)

            # Taxi decision-making loop
            while not self.goal_finished:
                result = self.choice()

                if not result:
                    break

            final_reward = self.calculate_final_reward(current_grid)

            isBetter = False

            if final_reward > best_reward:
                # Increase learning rate by a lot
                # self.learning_rate = min(1.0, self.learning_rate * 3)

                isBetter = True

                # self.learning_rate = min(self.initial_learning_rate, self.learning_rate + self.learning_rate * (final_reward - best_reward) / 100)
                self.learning_rate = self.initial_learning_rate

                best_reward = final_reward

            self.calculate_discounted_rewards_and_update_q_table(final_reward)

            # Print episode details
            # If finished color green, else color red
            
            color = Fore.WHITE
            if (self.goal_finished):
                color = Fore.YELLOW
                if (isBetter):
                    color = Fore.GREEN
            else:
                color = Fore.RED
                if (isBetter):
                    color = Fore.MAGENTA
            

            print(color + f"Episode: {episode}, Moves = {self.current_turn}, Final Reward = {final_reward}, Epsilon = {self.epsilon}, Learning Rate = {self.learning_rate}")





def print_grid(grid):
    for col in grid[0]:
        print(' --- ', end='')
    print("")
    for row in grid:
        print(f"{row}")
    for col in grid[0]:
        print(' --- ', end='')
    print("")


def get_pos_value(grid, pos):
    return grid[pos[0]][pos[1]]


def add_value_to_pos(grid, pos, value):
    grid[pos[0]][pos[1]] = value

def find_pos_of_value(grid, value):
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == value:
                return (i, j)
    return None

# Function to get the state based on agent's position and surroundings
def get_state(grid):
    grid = grid_qtable(grid)
    # print(grid)
    return (grid)


# Function to convert the entire grid state to a string for Q-table lookup
def grid_to_string(grid):
    state_str = ''.join([''.join(row) for row in grid])
    return state_str

# Function to convert the relevant part of the grid state to a string for Q-table lookup
def grid_qtable(grid):
    # Find the positions of 'T' (taxi), 'G' (goal), and 'C' (customer) in the grid
    taxi_position = None
    goal_position = None
    customer_position = None

    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == 'T':
                taxi_position = (i, j)
            elif cell == 'G':
                goal_position = (i, j)
            elif cell == 'C':
                customer_position = (i, j)

    # Extract the 5x5 nearby cells around the taxi with out-of-bounds handling
    nearby_cells = []
    for i in range(taxi_position[0] - 2, taxi_position[0] + 3):
        row_cells = []
        for j in range(taxi_position[1] - 2, taxi_position[1] + 3):
            if 0 <= i < len(grid) and 0 <= j < len(grid[0]):
                row_cells.append(grid[i][j])
            else:
                row_cells.append('-1')  # Out-of-bounds cells
        nearby_cells.append(row_cells)
    
    # Convert the nearby cells to a string
    nearby_str = ''.join([''.join(row) for row in nearby_cells])

    # Get grid row length
    grid_row_length = len(grid[0])

    # Determine the quadrant of the taxi
    taxi_quadrant = (1 if taxi_position[0] < grid_row_length else -1,
                     1 if taxi_position[1] >= grid_row_length else -1)

    # Determine the quadrant of the goal
    if(goal_position is not None):
        goal_quadrant = (1 if goal_position[0] < grid_row_length else -1,
                     1 if goal_position[1] >= grid_row_length else -1)
    else: goal_quadrant = taxi_quadrant

    # Determine the quadrant of the customer or set it to (0, 0) if customer doesn't exist
    if(customer_position is not None):
        customer_quadrant = (0, 0) if customer_position is None else (1 if customer_position[0] < grid_row_length else -1,
                                                                  1 if customer_position[1] >= grid_row_length else -1)
    else: customer_quadrant = taxi_quadrant

    # Combine the nearby cells string and the quadrant information for taxi, goal, and customer
    state_str = f"{nearby_str}-T{taxi_quadrant}G{goal_quadrant}C{customer_quadrant}"

    return state_str


# Convert grid string to actual grid for study (mainly to calculate agent -> customer and agent -> goal distance)
def unpack_grid(grid_string, num_rows, num_columns):
    grid = []
    for i in range(num_columns):
        start = i * num_rows
        end = start + num_rows
        col_string = grid_string[start:end]
        col = list(col_string)
        grid.append(col)
    return grid


def are_grids_equal(grid1, grid2):
    if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
        return False  # Grids have different dimensions

    for row1, row2 in zip(grid1, grid2):
        if row1 != row2:
            return False  # Rows are not equal

    return True  # All rows are equal

def place_objects(grid, objects=['C', 'G']):
    available_positions = [(i, j) for i, row in enumerate(grid) for j, cell in enumerate(row) if cell == '0']

    for obj in objects:
        if not available_positions:
            break  # No more available positions

        position = random.choice(available_positions)
        grid[position[0]][position[1]] = obj
        available_positions.remove(position)

    return grid

# Q-Learning parameters

# First Agent Parameters
# Q_TABLE = {}  # Initialize an empty Q-table
LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.9
EPSILON1 = 0.4
EPSILON2 = 0.5

# Pickle file for Q1
agent1_save = "Agent1.pickle"
agent1_m1_save = "Agent1_M1.pickle"
agent1_m2_save = "Agent1_M2.pickle"
agent1_m3_save = "Agent1_M3.pickle"

AGENT_1_SAVES = [agent1_m1_save, agent1_m2_save, agent1_m3_save]


# Pickle file for Q2
agent2_save = "Agent2.pickle"
agent2_m1_save = "Agent2_M1.pickle"
agent2_m2_save = "Agent2_M2.pickle"
agent2_m3_save = "Agent2_M3.pickle"

AGENT_2_SAVES = [agent2_m1_save, agent2_m2_save, agent2_m3_save]

# Define constants
MAX_MOVES = 50
NUM_EPISODES = 200000
ACTIONS = ['up', 'right', 'down', 'left']

# Graph constants
GROUP_BY = int(NUM_EPISODES/5)

# Maps

MAP_4 = [
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', 'G', '0', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '1', '1', '1', '1', '1', '1', '1', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '1'],
    ['1', 'C', '0', '0', '0', '0', '0', '0', '0', '1', '1', '0', '0', '0', '1'],
    ['1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '1', '1', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '1'],
    ['1', '0', '0', '1', '1', '1', '0', '0', '0', '0', '0', '1', '1', '0', '1'],
    ['1', '0', 'T', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']
]


MAPS_BASE = [MAP_4]


# Training
taxi1 = TaxiAgent()  # RL Agent
taxi1.train_agent(MAPS_BASE[0], NUM_EPISODES)

    
    # # Save Q-Table
    # taxi1.save_Q_TABLE(AGENT_1_SAVES[i])


    # # Training Agent 2
    # taxi2 = TaxiAgent(qtable={}, epsilon=EPSILON2)  # RL Agent
    # taxi2.set_training(True)
    # print("Starting agent 2 training...")
    # for episode in range(1,(NUM_EPISODES+1)):
    #     # Copy map grid
    #     if (episode % 100 == 0): print("Episode = ", episode)
    #     current_grid = copy.deepcopy(MAPS_VARIANCE[i])

    #     # Find taxi position
    #     taxi_pos = (0, 0)
    #     for j, row in enumerate(current_grid):
    #         if 'T' in row:
    #             taxi_pos = (j, row.index('T'))
    #             break

    #     taxi2.restart_round(taxi_pos, current_grid)

    #     # Taxi has finished
    #     while not taxi2.goal_finished:
    #         result = taxi2.choice()

    #         if not result:
    #             break

    #     final_reward = taxi2.calculate_final_reward(current_grid)
    #     taxi2.calculate_discounted_rewards_and_update_q_table(final_reward)

    # # Save Q-Table
    # taxi2.save_Q_TABLE(AGENT_2_SAVES[i])


TEST_EPISODES = 1

# Testing Map base

for i in range(len(MAPS_BASE)):
    print(f"Map: {i}")

    # Load agent 1
    
    # Load Q-Table
    # with open(AGENT_1_SAVES[i], 'rb') as f:
    #     load_qtable = pickle.load(f)

    # taxi1 = TaxiAgent(qtable={}, epsilon=EPSILON1)  # RL Agent
    taxi1.set_training(False)
    test_wins = 0
    for episode in range(TEST_EPISODES):
        current_grid = copy.deepcopy(MAPS_BASE[i])
        taxi_pos = (0, 0)
        for j, row in enumerate(current_grid):
            if 'T' in row:
                taxi_pos = (j, row.index('T'))
                break

        taxi1.restart_round(taxi_pos, current_grid)

        moves = 0
        while not taxi1.goal_finished:
            print_grid(current_grid)   
            result = taxi1.choice()
            moves += 1

            if not result:
                break

        if taxi1.goal_finished:
            test_wins += 1

    print(f'Agent 1 Test wins: {test_wins} with moves = {moves}')

    # Load agent 2

    # Load Q-Table
    # with open(AGENT_1_SAVES[i], 'rb') as f:
    #     load_qtable = pickle.load(f)

    # taxi2 = TaxiAgent(qtable=load_qtable, epsilon=EPSILON2)  # RL Agent
    # taxi2.set_training(False)
    # test_wins = 0
    # for episode in range(TEST_EPISODES):
    #     current_grid = copy.deepcopy(MAPS_BASE[i])
    #     taxi_pos = (0, 0)
    #     for j, row in enumerate(current_grid):
    #         if 'T' in row:
    #             taxi_pos = (j, row.index('T'))
    #             break

    #     taxi2.restart_round(taxi_pos, current_grid)

    #     moves = 0
    #     while not taxi2.goal_finished:
    #         # print_grid(current_grid)   
    #         result = taxi2.choice()
    #         moves += 1

    #         if not result:
    #             break

    #     if taxi2.goal_finished:
    #         test_wins += 1

    # print(f'Agent 2 Test wins: {test_wins} with moves = {moves}')