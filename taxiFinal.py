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
from heapq import heappop, heappush
import time
import pandas as pd
from IPython.display import display


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

    def __init__(self, qtable={}, state_type = 'alt', limitQ = True, initial_epsilon=0.3, 
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
        self.state_type = state_type
        self.limitQ = limitQ

    def restart_round(self, starting_location, grid):
        super().restart_round(starting_location, grid)
        self.episode_rewards = 0
        self.transitions = []
        self.episode += 1

        # Adjust epsilon and learning rate based on performance
        if self.episode_rewards > self.best_reward:
            self.best_reward = self.episode_rewards
        else:
            self.learning_rate = max(self.min_learning_rate, self.learning_rate * self.learning_rate_decay)
        
        self.recent_rewards.append(self.episode_rewards)

    def store_transition(self, state, action):
        self.transitions.append((state, action))

    def calculate_discounted_rewards_and_update_q_table(self, final_reward):
        cumulative_reward = final_reward

        # Iterate through the transitions in reverse order
        for state, action in (self.transitions):
            currentAction = action
            current_q_value = self.qtable[state][action]
            new_q_value = current_q_value + self.learning_rate * (cumulative_reward - current_q_value)
            # Clip the Q-value within the specified bounds
            if (self.limitQ):
                new_q_value = max(MIN_Q_VALUE, min(new_q_value, MAX_Q_VALUE))

            self.qtable[state][action] = new_q_value
            cumulative_reward = DISCOUNT_FACTOR * cumulative_reward

            if (self.limitQ):

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
                
                # If any action is less than min value, then set it to min value
                min_actions = [action for action, value in self.qtable[state].items() if value <= MIN_Q_VALUE]
                if len(min_actions) > 0:
                    for action in min_actions:
                        self.qtable[state][action] = MIN_Q_VALUE

    def get_q_value(self, state, action):
        return self.qtable.get((state, action), 0)

    def calculate_immediate_reward(self, current_grid, previous_grid, action):
        reward = 0
        
        # Check if the agent has reached the goal
        goal_pos = find_pos_of_value(current_grid, 'G')
        agent_pos = find_pos_of_value(current_grid, 'T')
        
        if goal_pos:
            reward += 10  # Immediate reward for reaching the goal
            return reward  # Return immediately since the goal was reached
        
        # Punish agent if it stayed in the same place
        if current_grid == previous_grid:
            reward -= 1/MAX_MOVES  # Small penalty for not moving
        
        # Calculate the Manhattan distance to the goal
        if not goal_pos:
            previous_agent_pos = find_pos_of_value(previous_grid, 'T')
            previous_distance = abs(goal_pos[0] - previous_agent_pos[0]) + abs(goal_pos[1] - previous_agent_pos[1])
            current_distance = abs(goal_pos[0] - agent_pos[0]) + abs(goal_pos[1] - agent_pos[1])
            
            # Reward for moving closer to the goal
            if current_distance < previous_distance:
                reward += 1/MAX_MOVES  # Small reward for moving closer
            else:
                reward -= 1/MAX_MOVES  # Small penalty for moving further away or not improving
        
        return reward

    def calculate_final_reward(self, grid):
        goal_exists = any('G' in row for row in grid)

        reward = 0

        # If goal is not on the grid, add +20 to the reward
        if not goal_exists:
            reward += 10
            reward += (MAX_MOVES - self.current_turn)/MAX_MOVES
        
        # Punish agent if it stayed in the same place\
        last_state = ''
        for i, transition in enumerate(self.transitions):
            state = transition[0]
            if state == last_state:
                reward -= 1/MAX_MOVES

        
        # If goal exists give reward based on distance (closer is better)
        if goal_exists:
            goal_pos = find_pos_of_value(grid, 'G')
            agent_pos = find_pos_of_value(grid, 'T')
            distance = abs(goal_pos[0] - agent_pos[0]) + abs(goal_pos[1] - agent_pos[1])
            # reward += (len(grid)*2 - distance) * math.sqrt(len(grid))
            reward += ((len(grid)*2 - distance) * 10)/MAX_MOVES

        # Subtract the number of turns taken by the agent
        reward -= self.current_turn/MAX_MOVES

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
    
    def get_best_action_probability(self, state):
        if state not in self.qtable:
            # If the state is not in the Q-Table, create a key with 0 values for all possible actions
            self.qtable[state] = {a: 0 for a in ACTIONS}

        if self.is_training:
            # Convert Q-values to a probability distribution using softmax
            q_values = np.array(list(self.qtable[state].values()))
            probabilities = np.exp(q_values) / np.sum(np.exp(q_values))
            
            # Select an action based on the probabilities
            action = np.random.choice(ACTIONS, p=probabilities)
            return action
        else:
            # Exploitation: choose the best action
            best_action = max(self.qtable[state], key=self.qtable[state].get)
            return best_action

    def choice(self):
        self.current_turn += 1
        if self.current_turn >= MAX_MOVES:
            return False

        # Get the current state based on the agent's position and surroundings
        state = get_state(self.grid, self.state_type)
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
        movesTaken = 0

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
            movesTaken += 1
            if storeResults:
                grids.append(copy.deepcopy(current_grid))

            if not result:
                break

        final_reward = self.calculate_final_reward(current_grid)    
        
        if training:
            self.calculate_discounted_rewards_and_update_q_table(final_reward)

        if storeResults:
            return final_reward, grids, current_grid
        else:
            return final_reward, movesTaken, current_grid
        
    
    def train_agent(self, map, num_episodes, ep_per_test=500):
        self.set_training(True)
        print("Starting agent training...")
        best_reward = float('-inf')
        average_reward = []
        average_moves = []

        # Difference from A*
        path = find_path(map)
        a_moves = len(path) - 1

        episodesPerTest = ep_per_test

        df = init_dataframe()

        for episode in range(1, (num_episodes + 1)):
            final_reward, _, _1 = self.run_map(map, training=True)
            
            average_reward.append(final_reward)
            if len(average_reward) > episodesPerTest:
                average_reward.pop(0)
            
            average_moves.append(self.current_turn)
            if len(average_moves) > episodesPerTest:
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

            if episode % episodesPerTest == 0:

                test_rewards, test_moves, currentGrid = self.run_map(map, training=False, storeResults=False)
                difError = abs(a_moves - test_moves) - 1

                movesToGoal = 0
                # If difError != 0, then we are not at the goal
                if difError != 0:
                    aCurrentPath = find_path(currentGrid)
                    movesToGoal = len(aCurrentPath)
                
                # Add df row
                # df = pd.concat([df, pd.DataFrame({'episode': [episode], 'avg_rewards': [avg_reward_value], 'test_rewards': [test_rewards], 'test_moves': [test_moves], 'difError': [difError], 'movesToGoal': [movesToGoal]})])
                df.loc[len(df.index)] = [episode, avg_reward_value, test_rewards, test_moves, difError, movesToGoal]
                # display(df)

                print("Epoch: ", episode)

        return df


def init_dataframe():
    return pd.DataFrame(columns=['episode', 'avg_rewards', 'test_rewards', 'test_moves', 'difError', 'movesToGoal'])

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
def get_state(grid, state_type):
    if (state_type == 'quadrant'):
        grid = grid_qtable(grid)
    elif (state_type == 'alt'):
        grid = grid_qtable_alt(grid)
    # print(grid)
    return (grid)

# Function to convert the relevant part of the grid state to a string for Q-table lookup
def grid_qtable(grid):
    # Find the positions of 'T' (taxi), 'G' (goal)
    taxi_position = None
    goal_position = None

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

def grid_qtable_alt(grid):
    # Find the positions of 'T' (taxi), 'G' (goal)
    taxi_position = None
    goal_position = None

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
    
    # Find the relative direction of the goal from the taxi (N, E, S, W, NE, SE, SW, NW)
    goal_direction = ""
    if goal_position:
        dy = goal_position[0] - taxi_position[0]
        dx = goal_position[1] - taxi_position[1]

        if dy < 0:
            goal_direction += "N"
        elif dy > 0:
            goal_direction += "S"

        if dx > 0:
            goal_direction += "E"
        elif dx < 0:
            goal_direction += "W"

    state_str = f"{nearby_str}_{goal_direction}"

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


# A* Algorithm

def manhattan_distance(start, goal):
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

def find_neighbors(node, grid):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    neighbors = []
    for direction in directions:
        neighbor = (node[0] + direction[0], node[1] + direction[1])
        if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]):
            if grid[neighbor[0]][neighbor[1]] != '1':  # Check if it's not a wall
                neighbors.append(neighbor)
    return neighbors

def a_star(grid, start, goal):
    open_set = []
    heappush(open_set, (0 + manhattan_distance(start, goal), 0, start))
    came_from = {}
    g_score = {node: float('inf') for node in np.ndindex(np.array(grid).shape)}
    g_score[start] = 0

    while open_set:
        current = heappop(open_set)[2]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Reverse path

        for neighbor in find_neighbors(current, grid):
            tentative_g_score = g_score[current] + 1  # Assuming cost is 1 for each move
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + manhattan_distance(neighbor, goal)
                heappush(open_set, (f_score, tentative_g_score, neighbor))
    return []  # If no path is found

def find_taxi_and_goal(grid):
    start = goal = None
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 'T':
                start = (i, j)
            elif grid[i][j] == 'G':
                goal = (i, j)
    return start, goal

def find_path(grid):
    start, goal = find_taxi_and_goal(grid)
    if not start or not goal:
        return []
    return a_star(grid, start, goal)


def train_agent(agent, map, agentName='Agent'):
    print(f"Training {agentName}...")
    start_time_train = time.time()

    train_dataframe = agent.train_agent(map, NUM_EPISODES, 10)

    end_time_train = time.time()
    print("Training time: " + str(end_time_train - start_time_train))

    return train_dataframe

def display_test(agent, rewards, grids, moves, agentName='Agent'):
    qValues = []

    for i in range(len(grids)):
        # print_grid(grids[i])
        state = get_state(grids[i], agent.state_type)
        # print(state)
        try:
            qValues.append(agent.qtable[state])
        except:
            print(qValues.append("Goal Reached"))
    
    print("Moves: " + str(moves))

    # Create a unique folder for this test run
    test_run_id = str(uuid.uuid4())
    folder_path = f'testrun-{agentName}-{test_run_id}'
    print("Test run id: ", test_run_id)

    image_paths = save_grids_as_images(grids, folder_path)

    # Initialize the GUI
    root = tk.Tk()
    root.title("Grid Viewer")
    app = GridViewer(root, image_paths, qValues)
    root.mainloop()


# Running


# Pickle file for Q1
agent1_save = "AgentDirectional.pickle"

# Agents (with directional/quadrant state and limit/no limit Q-table)
agent_dir_limit_save = "AgentDirectional_Limit.pickle"
agent_quad_limit_save = "AgentQuadrant_Limit.pickle"
agent_dir_no_limit_save = "AgentDirectional_NoLimit.pickle"
agent_quad_no_limit_save = "AgentQuadrant_NoLimit.pickle"

# Define constants
MAX_MOVES = 100
NUM_EPISODES = 20000
DISCOUNT_FACTOR = 0.999
MAX_Q_VALUE = 20
MIN_Q_VALUE = -1
ACTIONS = ['up', 'right', 'down', 'left']

# Graph constants
GROUP_BY = int(NUM_EPISODES/5)

# Maps

MAP_1 = [
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
    ['1', '0', '0', '0', '0', '0', '1', '0', '0', '0', '1', '0', '0', 'G', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '1', '0', '0', '0', '1', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '1', '1', '1', '1', '1', '1', '1', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', 'T', '0', '0', '0', '1', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '0', '0', '0', '1'],
    ['1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '1', '1', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '1'],
    ['1', '0', '0', '1', '1', '1', '0', '0', '0', '0', '0', '1', '1', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']
]

MAP_2 = [
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
    ['1', 'T', '0', '0', '0', '1', 'G', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '1', '1', '0', '0', '1', '1', '1', '1', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '1', '0', '0', '0', '0', '0', '1', '1', '1'],
    ['1', '0', '1', '1', '1', '0', '1', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '1', '0', '0', '0', '1', '0', '0', '1', '0', '0', '0', '0', '1'],
    ['1', '0', '1', '0', '1', '1', '1', '0', '0', '1', '0', '0', '0', '0', '1'],
    ['1', '0', '1', '0', '0', '1', '0', '0', '0', '1', '1', '0', '0', '0', '1'],
    ['1', '0', '1', '1', '0', '1', '0', '0', '0', '0', '1', '1', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '1', '0', '0', '0', '0', '0', '1', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '1', '1', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']
]

MAP_1_ADVANCED  = [
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
    ['1', '0', 'T', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '1', '1', '1', '1', '1', '1', '1', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '0', '0', '0', '1'],
    ['1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '1', '1', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '1'],
    ['1', '0', '0', '1', '1', '1', '0', '0', '0', '0', '0', '1', '1', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '1', '1', '1', '1', '1', '1', '1', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', 'G', '0', '0', '1', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '0', '0', '0', '1'],
    ['1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '1', '1', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '1'],
    ['1', '0', '0', '1', '1', '1', '0', '0', '0', '0', '0', '1', '1', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']
]


MAPS_BASE = [MAP_1, MAP_1_ADVANCED, MAP_2]

# taxiPlayer = TaxiDriver()
# taxiPlayer.play(MAP_1)

# Training
# start_time_train = time.time()

# Load pickle if it exists
# if os.path.exists(agent1_save):
#     with open(agent1_save, 'rb') as f:
#         qTable = pickle.load(f)
# else:
#     qTable = {}

# taxi1 = TaxiAgent(qTable)  # RL Agent
# taxi1.train_agent(MAPS_BASE[1], NUM_EPISODES)

# Agents
agent_dir_limit = TaxiAgent({}, 'alt', True)  # RL Agent
agent_dir_no_limit = TaxiAgent({}, 'alt', False)
agent_quad_limit = TaxiAgent({}, 'quadrant', True)
agent_quad_no_limit = TaxiAgent({}, 'quadrant', False)


train_df = train_agent(agent_dir_limit, MAPS_BASE[0], 'Agent Directional with Q Limit')
print("Training DF")
display(train_df)

rewards, grids, _ = agent_dir_limit.run_map(MAPS_BASE[0], training=False, storeResults=True)
display_test(agent_dir_limit, rewards, grids, len(grids), 'Agent Directional with Q Limit')

# Create a plot based on episodes and avg_rewards

# Assuming train_df is already defined and has the necessary columns
episodes = train_df['episode']

# Custom settings for a similar visual look
marker_style = dict(marker='.', markersize=5, linestyle='-', linewidth=1.5)

window_size = 100  # You can adjust this size to make the plot smoother or less smooth

# Calculate the moving average of the average rewards
smoothed_rewards = train_df['test_rewards'].rolling(window=window_size).mean()

# Plot 1: episodes vs test_rewards
plt.figure(figsize=(12, 6))
plt.plot(episodes, train_df['test_rewards'], **marker_style)
plt.xlabel('Episode')
plt.ylabel('Average Rewards')
plt.title('Episodes vs Average Rewards')
plt.grid(True)
plt.show()

# Plot 2: episodes vs difError
plt.figure(figsize=(12, 6))
plt.plot(episodes, train_df['difError'], **marker_style)
plt.xlabel('Episode')
plt.ylabel('DifError')
plt.title('Episodes vs DifError')
plt.grid(True)
plt.show()


# end_time_train = time.time()
# print("Training time: " + str(end_time_train - start_time_train))

# # Save qTable
# with open(agent1_save, 'wb') as f:
#     pickle.dump(agent_dir_limit.qtable, f)
