import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy


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

    def __init__(self):
        super().__init__()
        self.transitions = []
        self.episode_rewards = 0
        self.is_training = True

    def restart_round(self, starting_location, grid):
        super().restart_round(starting_location, grid)
        self.episode_rewards = 0
        self.transitions = []

    def store_transition(self, state, action):
        self.transitions.append((state, action))

    def calculate_discounted_rewards_and_update_q_table(self, final_reward):
        cumulative_reward = final_reward

        # Iterate through the transitions in reverse order
        for state, action in reversed(self.transitions):
            Q_TABLE[state][action] += LEARNING_RATE * cumulative_reward
            cumulative_reward = DISCOUNT_FACTOR * cumulative_reward

    def get_q_value(self, state, action):
        return Q_TABLE.get((state, action), 0)

    def calculate_final_reward(self, grid):
        customer_exists = any('C' in row for row in grid)
        goal_exists = any('G' in row for row in grid)

        reward = 0

        # If customer is not on the grid, add +10 to the reward
        if not customer_exists:
            reward += 10

        # If goal is not on the grid, add +20 to the reward
        if not goal_exists:
            reward += 20

        # Subtract the number of turns taken by the agent
        reward -= self.current_turn

        return reward

    def get_best_action(self, state):

        if state not in Q_TABLE:
            # If the state is not in the Q-Table, create a key with 0 values for all possible actions
            Q_TABLE[state] = {a: 0 for a in ACTIONS}

        if ((random.random() < EPSILON) and self.is_training):
            # Exploration: with probability epsilon, choose a random action
            return random.choice(ACTIONS)
        else:
            best_action = max(Q_TABLE[state], key=Q_TABLE[state].get)
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

    # Determine the quadrant of the taxi
    taxi_quadrant = (1 if taxi_position[0] < 4 else -1,
                     1 if taxi_position[1] >= 4 else -1)

    # Determine the quadrant of the goal
    if(goal_position is not None):
        goal_quadrant = (1 if goal_position[0] < 4 else -1,
                     1 if goal_position[1] >= 4 else -1)
    else: goal_quadrant = taxi_quadrant

    # Determine the quadrant of the customer or set it to (0, 0) if customer doesn't exist
    if(customer_position is not None):
        customer_quadrant = (0, 0) if customer_position is None else (1 if customer_position[0] < 4 else -1,
                                                                  1 if customer_position[1] >= 4 else -1)
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
Q_TABLE = {}  # Initialize an empty Q-table
LEARNING_RATE = 0.8
DISCOUNT_FACTOR = 0.9
EPSILON = 0.3
NUM_EPISODES = 300
NUM_MAPS = 10

# Define constants
GRID_SIZE = 7
MAX_WALLS = 5
MAX_MOVES = 50
ACTIONS = ['up', 'right', 'down', 'left']

# Graph constants
GROUP_BY = int(NUM_EPISODES/5)

# xfile = openpyxl.load_workbook('TaxiScoresGridRevealedWalls2.xlsx')
# sheet = xfile.get_sheet_by_name('Sheet1')

# taxi = TaxiDriver()  # Human Player
wins = 0
current_episode = 1

turns = []
c_picked = []
rewards = []

BASE_GRID = [
        ['1', '1', '1', '1', '1', '1', '1', '1'],
        ['1', '0', '0', '0', '0', '0', '0', '1'],
        ['1', '1', '0', '0', '1', '0', '0', '1'],
        ['1', '1', '0', '1', '1', '0', '1', '1'],
        ['1', '0', '0', '0', '0', '0', '0', '1'],
        ['1', '0', '1', '1', '0', 'T', '0', '1'],
        ['1', '0', '0', '0', '0', '1', '0', '1'],
        ['1', '1', '1', '1', '1', '1', '1', '1']
]

# current_grid = [
#     ['1', '1', '1', '1', '1', '1'],
#     ['1', '0', '0', 'G', '0', '1'],
#     ['1', 'C', '1', '0', '0', '1'],
#     ['1', '0', '0', 'T', '0', '1'],
#     ['1', '0', '0', '0', '0', '1'],
#     ['1', '1', '1', '1', '1', '1'],
# ]
# print(f"Map: {map}, Episode: {episode}")

# Load AI Trained

with open("Taxi-New.pickle", "rb") as f:
    Q_TABLE = pickle.load(f)

test_wins = 0
taxi = TaxiAgent()  # RL Agent
for map in range(NUM_MAPS):
    print(f"Start Map {map} training")
    temp_grid = copy.deepcopy(BASE_GRID)
    map_grid = place_objects(temp_grid, objects=['C', 'G'])
    test_map = copy.deepcopy(map_grid)
    wins=0
    for episode in range(NUM_EPISODES):
        current_grid = copy.deepcopy(map_grid)
        #restart_game(taxi)
        taxi_pos = (0, 0)
        for i, row in enumerate(current_grid):
            if 'T' in row:
                taxi_pos = (i, row.index('T'))
                break

        taxi.restart_round(taxi_pos, current_grid)

        while not taxi.goal_finished:
            result = taxi.choice()

            if not result:
                break

        final_reward = taxi.calculate_final_reward(current_grid)
        # print(final_reward)
        taxi.calculate_discounted_rewards_and_update_q_table(final_reward)
        # For Excel data
        c_picked.append(taxi.c_picked)
        turns.append(taxi.current_turn)
        rewards.append(taxi.episode_rewards)
        # print(episode)
        if taxi.goal_finished:
            # print("won")
            wins += 1

    # Testing Run without traning
    print(f"Test map {map} run")
    taxi.is_training = False
    current_grid = test_map

    taxi_pos = (0, 0)
    for i, row in enumerate(current_grid):
        if 'T' in row:
            taxi_pos = (i, row.index('T'))
            break

    taxi.restart_round(taxi_pos, current_grid)

    while not taxi.goal_finished:
        print_grid(current_grid)    
        result = taxi.choice()

        if not result:
            break    

    if taxi.goal_finished:
        print("won")
        test_wins += 1
    
    print(f'Training wins: {wins}')

print(f'Test wins: {test_wins}')
# for index in range(len(turns) - 1):
#     sheet['B' + str(index + 2)] = turns[index]
#     sheet['C' + str(index + 2)] = c_picked[index]
# xfile.save('TaxiScoresGridRevealedWalls2.xlsx')

# tmpTurns = []  # X Values
# tmpValues = []  # Y Values
# for i in range(0, len(turns), GROUP_BY):
#     tmpTurns.append(i + GROUP_BY)
#     tmpValues.append(np.mean(turns[i::i+GROUP_BY]))

# plt.plot(tmpTurns, tmpValues)
# plt.xlabel(f'Epochs\nLearning Rate = {LEARNING_RATE}, Epsilon = {EPSILON}')
# plt.ylabel('Moves to finish')
# plt.title('Agent Episodes Graph')

# plt.show()

# Save RL Q-Table of training
with open("Taxi-New.pickle", "wb") as f:
    pickle.dump(Q_TABLE, f)


