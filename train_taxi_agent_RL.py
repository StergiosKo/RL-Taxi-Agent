import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import os
import uuid
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Label, Button
from heapq import heappop, heappush
import time
import pandas as pd


class TaxiDriver:
    def __init__(self):
        self.position = [0, 0]
        self.grid = []
        self.goal_finished = False
        self.current_turn = 0
        self.c_picked = -1
        self.hit_wall = False
        self.customer_count = 0

    def restart_round(self, starting_location, grid):
        self.position = starting_location
        self.grid = grid
        self.goal_finished = False
        self.current_turn = 0
        self.c_picked = -1
        self.hit_wall = False
        self.customer_count = 0

    def move(self, direction):
        old_pos_value = '0'
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

        if new_pos_value == '1':
            # Mark the hit location with 'H' before ending the game
            add_value_to_pos(self.grid, new_pos, 'H')
            self.goal_finished = True
            self.hit_wall = True
            return False  # End the game

        if new_pos_value == 'G':
            self.drop_passenger()

        if new_pos_value == 'C':
            self.customer_count += 1  # Increase the customer count
            # When a customer is collected, the customer disappears (becomes empty space '0')
            old_pos_value = '0'

        # Update previous loc value
        add_value_to_pos(self.grid, self.position, old_pos_value)
        self.position = new_pos
        # Update current loc value
        add_value_to_pos(self.grid, self.position, 'T')

        return True

    def drop_passenger(self):
        self.goal_finished = True

    def choice(self):
        self.current_turn += 1
        if self.current_turn >= MAX_MOVES:
            return False
        print_grid(self.grid)
        choice = input(f"Choose an action (up/right/down/left): ").lower()
        if not self.move(choice):
            return False  # End the game if move returns False

        print_grid(self.grid)
        return True
    
    def check_if_wall_hit(self):
        # Check if H is in the map
        wall_hit = any('H' in row for row in self.grid)

        return wall_hit

    def play(self, grid, qtable=None):
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
            if qtable:
                # Print qvalues
                state = grid_qtable_alt(current_grid)
                print(qtable[state])
            result = self.choice()

            if not result or self.check_if_wall_hit():
                print_grid(current_grid)
                break



class TaxiAgent(TaxiDriver):

    def __init__(self, qtable={}, reward_type='delayed', exploration='greedy', limitQ = True, state_type = 'alt', initial_epsilon=0.3, min_epsilon=0.01, epsilon_decay=0.99995,
                 initial_learning_rate=0.01, min_learning_rate=0.001, learning_rate_decay=0.999995):
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
        self.initial_learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.episode = 0
        self.best_reward = float('-inf')
        self.recent_rewards = []
        self.turns_without_increasing = 0
        self.state_type = state_type
        self.limitQ = limitQ
        self.reward_type = reward_type
        self.exploration = exploration
        self.currentaction = None
        self.immediate_rewards_turns = 0

    def restart_round(self, starting_location, grid):
        super().restart_round(starting_location, grid)
        self.episode_rewards = 0
        self.transitions = []
        self.episode += 1
        self.immediate_rewards_turns = 0

        self.learning_rate = max(self.min_learning_rate, self.initial_learning_rate * self.learning_rate_decay ** self.episode)
        self.epsilon = max(self.min_epsilon, self.initial_epsilon * self.epsilon_decay ** self.episode)
        
        self.recent_rewards.append(self.episode_rewards)

    def store_transition(self, state, action, reward):
        self.transitions.append((state, action, reward))

    def calculate_discounted_rewards_and_update_q_table(self):
        # Assume self.transitions includes (state, action, reward) tuples
        # First, calculate returns from each state to the end of the episode
        returns = []
        G = 0
        # Reverse the transitions as Monte Carlo updates go from end to start
        for state, action, reward in reversed(self.transitions):
            G = reward + DISCOUNT_FACTOR * G
            returns.insert(0, (state, action, G))  # Prepend to keep the order of states

        # Now, update the Q-values based on the returns calculated
        for state, action, G in returns:
            current_q_value = self.qtable[state][action]
            new_q_value = (1 - self.learning_rate) * current_q_value + self.learning_rate * G
            self.update_q_values(state, action, new_q_value)

        self.transitions = []  # Reset transitions after processing

    def get_q_value(self, state, action):
        return self.qtable.get((state, action), 0)

    def update_q_values(self, state, current_action, q_value):
        if (self.limitQ):
            q_value = max(MIN_Q_VALUE, min(q_value, MAX_Q_VALUE))
        
        try:
            self.qtable[state][current_action] = q_value
        except:
            self.qtable[state] = {a: 0 for a in ACTIONS}
            self.qtable[state][current_action] = q_value
        if (self.limitQ):
            # Ensure not all actions of this state have the max value
            max_actions = [action for action, value in self.qtable[state].items() if value >= MAX_Q_VALUE]

            # If 1 action has the max value, then subtract the learning rate of all the other actions
            # If 2 or more actions have the max value, then subtract the learning rate of these actions except the current action
            if len(max_actions) == 1:
                for action in self.qtable[state]:
                    if action != current_action:
                        self.qtable[state][action] -= self.learning_rate
            elif len(max_actions) > 1:
                for action in self.qtable[state]:
                    if (action != current_action) and (action not in max_actions):
                        self.qtable[state][action] -= self.learning_rate * 0.1
            
            # Do the same for min values
            min_actions = [action for action, value in self.qtable[state].items() if value <= MIN_Q_VALUE]
            if len(min_actions) == 1:
                for action in self.qtable[state]:
                    if action != current_action:
                        self.qtable[state][action] += self.learning_rate
            elif len(min_actions) > 1:
                for action in self.qtable[state]:
                    if (action != current_action) and (action not in min_actions):
                        self.qtable[state][action] += self.learning_rate * 0.1
            
            # If any action is less than min value, then set it to min value
            min_actions = [action for action, value in self.qtable[state].items() if value <= MIN_Q_VALUE]
            if len(min_actions) > 0:
                for action in min_actions:
                    self.qtable[state][action] = MIN_Q_VALUE

    
    # For customer immediate dinstance and customer rewards
    def calculate_immediate_reward(self, previous_grid, current_grid):
        reward = 0
        
        # Check if the agent has reached the goal
        goal_pos = find_pos_of_value(current_grid, 'G')
        agent_pos = find_pos_of_value(current_grid, 'T')

        # Get amount of customers in grid
        prev_customer_count = sum(x.count('C') for x in previous_grid)
        new_customer_count = sum(x.count('C') for x in current_grid)

        if new_customer_count < prev_customer_count:
            reward += 2
            # print("got customer")
        
        
        if self.hit_wall:
            reward -= 1
        
        if not goal_pos:
            reward += 20 - min(self.current_turn, 15)  # Immediate reward for reaching the goal
            return reward  # Return immediately since the goal was reached
        
        # For big map
        # if not goal_pos:
        #     reward += 50 - min(self.current_turn, 45)  # Immediate reward for reaching the goal
        #     return reward  # Return immediately since the goal was reached
        
        # # Small penalty for unnecessary moves
        reward -= 0.0001
        
        
        return reward
    

    def get_best_action(self, state):

        if state not in self.qtable:
            # If the state is not in the Q-Table, create a key with 0 values for all possible actions
            self.qtable[state] = {a: 0 for a in ACTIONS}

        if ((random.random() < self.epsilon) and self.is_training):
            # Exploration: with probability epsilon, choose a random action
            return random.choice(ACTIONS)
        else:

            # Check if there are multiple actions with the same highest Q-value in key=self.qtable[state].get
            best_actions = [action for action, value in self.qtable[state].items() if value == max(self.qtable[state].values())]

            # If there are multiple actions with the same highest Q-value, choose one of them at random, otherwise choose the best action
            if len(best_actions) > 1:
                return random.choice(best_actions)
            else:
                return best_actions[0]
            

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
        if self.exploration == 'greedy':
            action = self.get_best_action(state)
        elif self.exploration == 'softmax':
            action = self.get_best_action(state)
        self.currentaction = action

        previous_grid = copy.deepcopy(self.grid)

        self.move(action)

        reward = self.calculate_immediate_reward(previous_grid, self.grid)

        self.store_transition(state, action, reward)

        return True
    
    def update_immediate_reward(self, last_grid, current_grid, new_reward):
        old_state = grid_qtable_alt(last_grid)
        old_state_reward = 0
        try:
            old_state_reward = self.qtable[old_state][self.currentaction]
        except:
            old_state_reward = 0
        new_state = grid_qtable_alt(current_grid)
        try:
            # Get max reward of new state
            max_reward = max(self.qtable[new_state].values())
        except:
            max_reward = 0
        added_reward = old_state_reward + self.learning_rate * (new_reward + DISCOUNT_FACTOR * max_reward - old_state_reward)
        self.update_q_values(old_state, self.currentaction, added_reward)

    def save_Q_TABLE(self, filename='qtable.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.qtable, f)
    
    def set_training(self, bool=True):
        self.is_training = bool
    
    def run_map(self, map, training=True, storeResults = False):
        self.set_training(training)
        current_grid = copy.deepcopy(map)
        movesTaken = 0

        sum_immediate_reward = 0

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
            last_grid = copy.deepcopy(current_grid)
            result = self.choice()
            movesTaken += 1

            if storeResults:
                grids.append(copy.deepcopy(current_grid))
            
            action_reward = self.calculate_immediate_reward(last_grid, current_grid)
            sum_immediate_reward += action_reward

            if self.reward_type == 'immediate':
                self.update_immediate_reward(last_grid, current_grid, action_reward)

            if not result:
                break
    
            
        if self.reward_type == 'delayed':
            # final_reward = self.calculate_final_reward(current_grid)
            final_reward = sum_immediate_reward
            # print(f"Final Reward: {final_reward}")
        elif self.reward_type == 'immediate':
            final_reward = sum_immediate_reward
            # print(f"Final Reward: {sum_immediate_reward}")
        
        if training and self.reward_type == 'delayed':
            self.calculate_discounted_rewards_and_update_q_table()
        
        if result and not self.hit_wall and self.current_turn < MAX_MOVES:
            new_reward = (50 - min(self.current_turn, 45))
            # final_reward += new_reward
        
        if training and self.reward_type == 'immediate' and result and not(self.hit_wall) and self.current_turn < MAX_MOVES:
            state = grid_qtable_alt(current_grid)
            try:
                self.update_q_values(state, self.currentaction, self.qtable[state][self.currentaction] + new_reward)
            except:
                self.qtable[state] = {a: 0 for a in ACTIONS}
                self.update_q_values(state, self.currentaction, new_reward)
        
        if storeResults:
            return final_reward, grids, current_grid
        else:
            return final_reward, movesTaken, current_grid
        
    
    def train_agent(self, map, num_episodes, divider=100):
        self.set_training(True)
        print("Starting agent training...")
        best_reward = float('-inf')
        best_batch_reward = float('-inf')

        episodesPerTest = num_episodes/divider

        print(f"Has limit? {self.limitQ}")

        df = init_dataframe()

        for episode in range(1, (num_episodes + 1)):
            final_reward, _, _1 = self.run_map(map, training=True)
            
            if final_reward > best_batch_reward:
                best_batch_reward = final_reward

            isBetter = False

            if final_reward > best_reward:
                # Increase learning rate by a lot
                # self.learning_rate = min(1.0, self.learning_rate * 3)

                isBetter = True

                # self.learning_rate = min(self.initial_learning_rate, self.learning_rate + self.learning_rate * (final_reward - best_reward) / 100)
                # self.learning_rate = self.initial_learning_rate

                best_reward = final_reward
                final_reward = round(final_reward, 1)
            
            if (isBetter):
                print(f"Episode: {episode}, Moves = {self.current_turn}, Final Reward = {final_reward}, Epsilon = {self.epsilon}, Learning Rate = {self.learning_rate}")

            if episode % episodesPerTest == 0:

                # Test run
                rewards, moves, last_grid =self.run_map(map, training=False)
                rewards = round(rewards, 1)

                # Check if goal exists in grid
                if self.goal_finished and not(self.hit_wall):
                    win = True
                    distance = 0
                else:
                    win = False
                    goal_pos = find_pos_of_value(last_grid, 'G')
                    agent_pos = find_pos_of_value(last_grid, 'T')
                    distance = abs(goal_pos[0] - agent_pos[0]) + abs(goal_pos[1] - agent_pos[1])
                
                # Get manhattan distance

                # print(f"Episode: {episode}, average moves: {avg_moves}, average reward: {avg_reward_value}, wins: {avg_wins}")
                print(f"Episode: {episode}, rewards = {rewards}, moves = {moves}, customers collected = {self.customer_count}, goal distance = {distance}")
                df.loc[len(df.index)] = [episode, rewards, moves, self.customer_count, distance]
                best_batch_reward = float('-inf')

        return df


def init_dataframe():
    return pd.DataFrame(columns=['episode', 'rewards', 'moves', 'customers', 'distance'])

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
        'G': (0, 255, 0),      # Goal - green
        'H': (255, 0, 0),      # Hit - red
        'C': (0, 0, 255)      # Customer - blue
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
            q_values_text = beautify_q_values(self.q_values_list[self.index])

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


def train_agent(agent, map, agentName='Agent', divider=100):
    print(f"Training {agentName}...")
    start_time_train = time.time()

    train_dataframe = agent.train_agent(map, NUM_EPISODES, divider)

    end_time_train = time.time()
    print("Training time: " + str(end_time_train - start_time_train))

    return train_dataframe

def display_test(agent, rewards, grids, moves, agentName='Agent', path=''):
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
    folder_path = f'{path}/agent_runs/testrun-{agentName}-{test_run_id}'
    print("Test run id: ", test_run_id)

    image_paths = save_grids_as_images(grids, folder_path)

    # Initialize the GUI
    root = tk.Tk()
    root.title("Grid Viewer")
    app = GridViewer(root, image_paths, qValues)
    root.mainloop()

def load_Q_table(agentName='Agent'):
    if os.path.exists(agentName):
        with open(agentName, 'rb') as f:
            return pickle.load(f)
    else:
        return {}

def save_Q_table(agent, agentName='Agent'):
    with open(agentName, 'wb') as f:
        pickle.dump(agent.qtable, f)


def plot_avg_metric(df, metric, window, label, color, log=False):
    episodes = df['episode']
    y = df[metric]
    
    average_y = []
    for ind in range(len(y) - window + 1):
        average_y.append(np.mean(y[ind:ind+window]))
    
    for ind in range(window - 1):
        average_y.insert(0, np.nan)
    
    # Check if log scale is requested
    if log:
        plt.yscale('log')

    plt.plot(episodes, average_y, color=color, label=label)

# Maps

MAP_CUSTOMER_MINI = [
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
    ['1', 'G', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1'],
    ['1', '1', '1', '1', '0', '0', '0', '0', '0', '1', '0', '1'],
    ['1', '0', '0', '1', '1', '0', '0', '0', '0', '1', '0', '1'],
    ['1', '0', '0', '0', '0', 'C', '0', '0', '1', '1', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '1', '1', '1', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', 'T', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']
]

MAP_CUSTOMER_MINI_NO_C = [
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
    ['1', 'G', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1'],
    ['1', '1', '1', '1', '0', '0', '0', '0', '0', '1', '0', '1'],
    ['1', '0', '0', '1', '1', '0', '0', '0', '0', '1', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '1', '1', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '1', '1', '1', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', 'T', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']
]

MAP_CUSTOMER_MINI_2 = [
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', 'G', '0', '0', '0', '0', '1'],
    ['1', '0', 'C', '0', '0', '0', '0', '1', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '1', '0', '0', '1'],
    ['1', '0', '0', '1', '1', '1', '1', '1', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', 'C', '0', '0', 'T', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']
]

MAP_CUSTOMER_MINI_3 = [
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', 'G', '0', '0', '0', '0', '1', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '1', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '1', '0', '0', '1'],
    ['1', '0', '0', '1', '1', '1', '1', '1', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', 'C', '1', '0', '0', '0', '0', 'T', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']
]

MAP_CUSTOMER_MINI_3_NO_C = [
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', 'G', '0', '0', '0', '0', '1', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '1', '0', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '1', '0', '0', '1'],
    ['1', '0', '0', '1', '1', '1', '1', '1', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '0', '0', '1', '0', '0', '0', '0', 'T', '0', '1'],
    ['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1'],
    ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']
]


MAPS_BASE = [MAP_CUSTOMER_MINI_3, MAP_CUSTOMER_MINI_3_NO_C, MAP_CUSTOMER_MINI, MAP_CUSTOMER_MINI_NO_C]

# Define constants
MAX_MOVES = 100
NUM_EPISODES = 50000
DISCOUNT_FACTOR = 0.99
MAX_Q_VALUE = 200
MIN_Q_VALUE = -1
ACTIONS = ['up', 'right', 'down', 'left']
HARD_LIMIT = 10000

# Graph constants
GROUP_BY = int(NUM_EPISODES/5)

agentName = 'Monte Carlo Punish 50k Map 1 true'
agentQTable = ''
mapName = 'Map3'
MAP_INDEX = 1

# Load Q table
qTable = load_Q_table(agentQTable + ".pickle")

# agent = TaxiAgent(qTable, reward_type='delayed', exploration='softmax', limitQ = False)
# agent = TaxiAgent(qTable, reward_type='delayed', exploration='greedy', limitQ = False, epsilon_decay=0.999995)

# agent = TaxiAgent(qTable, reward_type='immediate', exploration='softmax', limitQ = False)
agent = TaxiAgent(qTable, reward_type='immediate', exploration='greedy', initial_learning_rate=0.01, min_learning_rate=0.001, epsilon_decay=0.999995)

# Training
df_training = train_agent(agent, MAPS_BASE[MAP_INDEX], agentName, 2000)

# Save csv
df_training.to_csv('FinalResults/Customers/Final/' + mapName + '/csv/' + agentName + '.csv')

# Save Q table
save_Q_table(agent, agentName + '.pickle')

# Testing
rewards, grids, _ = agent.run_map(MAPS_BASE[MAP_INDEX], training=False, storeResults=True)
display_test(agent, rewards, grids, len(grids), agentName, 'FinalResults/Customers/Final/' + mapName)

taxiPlayer = TaxiDriver()
taxiPlayer.play(MAPS_BASE[MAP_INDEX], agent.qtable)