import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy
from colorama import Fore
import math
import os
import uuid
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Label, Button


class TaxiDriver:
    def __init__(self):
        self.position = [0, 0]
        self.grid = []
        self.goal_finished = False
        self.current_turn = 0
        self.c_picked = -1

    def restart_round(self, starting_location, grid):
        self.position = starting_location
        self.grid = grid
        self.goal_finished = False
        self.current_turn = 0
        self.c_picked = -1

    # If the new position the taxi moves has a goal and the taxi has a customer, finish
    def drop_passenger(self):
        self.goal_finished = True

    def move(self, direction):
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

        if new_pos_value == 'G':
            self.drop_passenger()

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

    def play(self, grid):
        current_grid = copy.deepcopy(grid)

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

class TaxiAgent(TaxiDriver):

    def __init__(self, qtable={}, initial_epsilon=0.3, 
                 initial_learning_rate=0.1, min_learning_rate=0.001, learning_rate_decay=0.99995):
        super().__init__()
        self.transitions = []
        self.episode_rewards = 0
        self.is_training = True
        self.qtable = copy.deepcopy(qtable)
        self.initial_epsilon = initial_epsilon
        self.initial_learning_rate = initial_learning_rate
        self.epsilon = initial_epsilon
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
            # self.epsilon = min(self.initial_epsilon, self.epsilon * 1.1)
            # self.learning_rate = min(self.initial_learning_rate, self.learning_rate * 1.1)
            self.turns_without_increasing = 0
        else:
            self.turns_without_increasing += 1
            # Normal decay
            # self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
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
            currentAction = action
            current_q_value = self.qtable[state][action]
            new_q_value = current_q_value + self.learning_rate * (cumulative_reward - current_q_value)
            # Clip the Q-value within the specified bounds
            new_q_value = max(MIN_Q_VALUE, min(new_q_value, MAX_Q_VALUE))
            self.qtable[state][action] = new_q_value
            cumulative_reward = DISCOUNT_FACTOR * cumulative_reward

            # Ensure not all actions of this state have the max value
            max_actions = [action for action, value in self.qtable[state].items() if value == MAX_Q_VALUE]

            # If 1 action has the max value, then subtract the learning rate of all the other actions
            # If 2 or more actions have the max value, then subtract the learning rate of these actions except the current action
            if len(max_actions) == 1:
                for action in self.qtable[state]:
                    if action != currentAction:
                        self.qtable[state][action] -= self.learning_rate
            elif len(max_actions) > 1:
                for action in self.qtable[state]:
                    if (action != currentAction) and (action not in max_actions):
                        self.qtable[state][action] -= self.learning_rate * (cumulative_reward - current_q_value)
            
            # Do the same for min values
            min_actions = [action for action, value in self.qtable[state].items() if value == MIN_Q_VALUE]
            if len(min_actions) == 1:
                for action in self.qtable[state]:
                    if action != currentAction:
                        self.qtable[state][action] += self.learning_rate
            elif len(min_actions) > 1:
                for action in self.qtable[state]:
                    if (action != currentAction) and (action not in min_actions):
                        self.qtable[state][action] += self.learning_rate * (cumulative_reward - current_q_value)

    def get_q_value(self, state, action):
        return self.qtable.get((state, action), 0)

    def calculate_final_reward(self, grid):
        goal_exists = any('G' in row for row in grid)

        reward = 0

        # If goal is not on the grid, add +20 to the reward
        if not goal_exists:
            reward += 50
            reward += (MAX_MOVES - self.current_turn)
        
        # Punish agent if it stayed in the same place\
        last_state = ''
        for i, transition in enumerate(self.transitions):
            state = transition[0]
            if state == last_state:
                reward -= 0.1

        
        # If goal exists give reward based on distance (closer is better)
        if goal_exists:
            goal_pos = find_pos_of_value(grid, 'G')
            agent_pos = find_pos_of_value(grid, 'T')
            distance = abs(goal_pos[0] - agent_pos[0]) + abs(goal_pos[1] - agent_pos[1])
            # reward += (len(grid)*2 - distance) * math.sqrt(len(grid))
            reward += (len(grid)*2 - distance) * 2

        # Subtract the number of turns taken by the agent
        reward -= self.current_turn

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
    
    def run_map(self, map, training=True, storeResults = False):
        self.set_training(training)
        current_grid = copy.deepcopy(map)

        grids = []
        if storeResults:
            grids.append(copy.deepcopy(current_grid))

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
            if storeResults:
                grids.append(copy.deepcopy(current_grid))

            if not result:
                break

        final_reward = self.calculate_final_reward(current_grid)    
        
        if training:
            self.calculate_discounted_rewards_and_update_q_table(final_reward)

        if storeResults:
            return final_reward, grids
        else:
            return final_reward
        
    
    def train_agent(self, map, num_episodes):
        self.set_training(True)
        print("Starting agent training...")
        best_reward = float('-inf')
        average_reward = []
        average_moves = []

        for episode in range(1, (num_episodes + 1)):
            final_reward = self.run_map(map, training=True)
            
            average_reward.append(final_reward)
            if len(average_reward) > 1000:
                average_reward.pop(0)
            
            average_moves.append(self.current_turn)
            if len(average_moves) > 1000:
                average_moves.pop(0)

            avg_reward_value = sum(average_reward) / len(average_reward)
            avg_reward_value = round(avg_reward_value, 1)

            avg_moves = sum(average_moves) / len(average_moves)
            avg_moves = round(avg_moves, 1)

            isBetter = False

            if final_reward > best_reward:
                # Increase learning rate by a lot
                # self.learning_rate = min(1.0, self.learning_rate * 3)

                isBetter = True

                # self.learning_rate = min(self.initial_learning_rate, self.learning_rate + self.learning_rate * (final_reward - best_reward) / 100)
                self.learning_rate = self.initial_learning_rate

                best_reward = final_reward

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
            
            if (isBetter):
                print(color + f"Episode: {episode}, Moves = {self.current_turn}, Final Reward = {final_reward}, Epsilon = {self.epsilon}, Learning Rate = {self.learning_rate}")

            if episode % 1000 == 0:
                test_rewards = self.run_map(map, training=False)
                color = Fore.WHITE
                if (self.goal_finished):
                    color = Fore.YELLOW
                    if (isBetter):
                        color = Fore.GREEN
                else:
                    color = Fore.RED
                    if (isBetter):
                        color = Fore.MAGENTA

                print(color + f"Episode = {episode}, Best Reward = {best_reward} Average Reward = {avg_reward_value} Test Reward = {test_rewards} Average Moves = {avg_moves}")


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
    taxi_quadrant = (1 if taxi_position[0] < grid_row_length/2 else -1,
                     1 if taxi_position[1] >= grid_row_length/2 else -1)

    # Determine the quadrant of the goal
    if(goal_position is not None):
        goal_quadrant = (1 if goal_position[0] < grid_row_length/2 else -1,
                     1 if goal_position[1] >= grid_row_length/2 else -1)
    else: goal_quadrant = taxi_quadrant

    # Combine the nearby cells string and the quadrant information for taxi, goal, and customer
    state_str = f"{nearby_str}-T{taxi_quadrant}G{goal_quadrant}"

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

def grid_to_image(grid, cell_size=20):
    # Define colors
    colors = {
        '1': (0, 0, 0),       # Wall - black
        '0': (255, 255, 255), # Empty space - white
        'T': (255, 255, 0),   # Taxi - yellow
        'G': (0, 255, 0)      # Goal - green
    }

    # Calculate image size
    height = len(grid)
    width = len(grid[0])
    img_height = height * cell_size
    img_width = width * cell_size

    # Create a new image with white background
    img = Image.new('RGB', (img_width, img_height), (255, 255, 255))
    pixels = img.load()

    # Fill the image with the grid colors
    for i in range(height):
        for j in range(width):
            color = colors[grid[i][j]]
            for y in range(i * cell_size, (i + 1) * cell_size):
                for x in range(j * cell_size, (j + 1) * cell_size):
                    pixels[x, y] = color

    return img

def save_grids_as_images(grids, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    image_paths = []
    for idx, grid in enumerate(grids):
        img = grid_to_image(grid, 50)
        img_path = os.path.join(folder_path, f'grid_{idx}.png')
        img.save(img_path)
        image_paths.append(img_path)

    return image_paths


class GridViewer:
    def __init__(self, master, image_paths, q_values_list):
        self.master = master
        self.image_paths = image_paths
        self.q_values_list = q_values_list
        self.index = 0

        self.label = Label(master)
        self.label.pack()

        self.info_label = Label(master, text="")
        self.info_label.pack()

        self.q_values_label = Label(master, text="", justify=tk.LEFT, wraplength=400)
        self.q_values_label.pack()

        self.prev_button = Button(master, text="Prev", command=self.prev_image)
        self.prev_button.pack(side="left")

        self.next_button = Button(master, text="Next", command=self.next_image)
        self.next_button.pack(side="right")

        self.update_image()

        # Bind arrow keys for navigation
        self.master.bind('<Left>', lambda event: self.prev_image())
        self.master.bind('<Right>', lambda event: self.next_image())

    def update_image(self):
        img_path = self.image_paths[self.index]
        img = Image.open(img_path)
        self.img_tk = ImageTk.PhotoImage(img)
        self.label.config(image=self.img_tk)
        self.info_label.config(text=f"Move {self.index + 1} out of {len(self.image_paths)}")

        # Use the beautify function if not the last image
        if self.index < len(self.image_paths) - 1:
            q_values_text = beautify_q_values(self.q_values_list[self.index])
        else:
            q_values_text = "No Q-values available for this state."

        self.q_values_label.config(text=q_values_text)

        self.prev_button.config(state=tk.NORMAL if self.index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.index < len(self.image_paths) - 1 else tk.DISABLED)

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.update_image()

    def next_image(self):
        if self.index < len(self.image_paths) - 1:
            self.index += 1
            self.update_image()



    
def beautify_q_values(q_values):
    # Define arrow symbols for directions
    arrows = {
        'up': '↑',
        'right': '→',
        'down': '↓',
        'left': '←'
    }

    # Format each Q-value to two decimal places and replace text with arrows
    formatted_q_values = ', '.join(f"{arrows[key]}: {value:.2f}" for key, value in q_values.items())
    return formatted_q_values






# Running





# Pickle file for Q1
agent1_save = "Agent1.pickle"

# Define constants
MAX_MOVES = 50
NUM_EPISODES = 15000
DISCOUNT_FACTOR = 0.9
MAX_Q_VALUE = 100
MIN_Q_VALUE = -100
ACTIONS = ['up', 'right', 'down', 'left']

# Graph constants
GROUP_BY = int(NUM_EPISODES/5)

# Maps

MAP_4 = [
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '1', '1', '1', '1', '1', '1', '1', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', 'G', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '0', '0', '0', '1'],
    ['1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '1', '1', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '1'],
    ['1', '0', '0', '1', '1', '1', '0', '0', '0', '0', '0', '1', '1', '0', '1'],
    ['1', '0', 'T', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']
]


MAPS_BASE = [MAP_4]

# taxiPlayer = TaxiDriver()
# taxiPlayer.play(MAP_4)

# Training

# Load pickle if it exists
if os.path.exists(agent1_save):
    with open(agent1_save, 'rb') as f:
        qTable = pickle.load(f)
else:
    qTable = {}

taxi1 = TaxiAgent(qTable)  # RL Agent
taxi1.train_agent(MAPS_BASE[0], NUM_EPISODES)

# Save qTable
with open(agent1_save, 'wb') as f:
    pickle.dump(taxi1.qtable, f)


TEST_EPISODES = 1
rewards, grids = taxi1.run_map(MAPS_BASE[0], training=False, storeResults=True)
moves = taxi1.current_turn

# For each grid, get Q-Values of the agent
qValues = []
for i in range(len(grids)):
    # print_grid(grids[i])
    state = grid_qtable(grids[i])
    try:
        qValues.append(taxi1.qtable[state])
    except:
        print(qValues.append("Goal Reached"))

print("Moves: " + str(moves))

# Create a unique folder for this test run
test_run_id = str(uuid.uuid4())
folder_path = f'testrun-{test_run_id}'
image_paths = save_grids_as_images(grids, folder_path)


# Initialize the GUI
root = tk.Tk()
root.title("Grid Viewer")
app = GridViewer(root, image_paths, qValues)
root.mainloop()