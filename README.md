<h1 align="center">Q-Learning in Grid Navigation</h1>
<p>A bachelor thesis project exploring Q-Learning agents in grid-based navigation environments, focusing on reward systems, state representations, and learning efficiency.</p>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-this-project">About This Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#installation-and-running">Installation and Running</a>
    <ul>
        <li><a href="#parameters">Parameters</a></li>
      </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THIS PROJECT -->
## About This Project
Reinforcement Learning Taxi Agent with Q-Learning, my thesis while studying on University of Macedonia, in bcs applied informatics.

This project was developed as part of a Bachelor’s thesis in Applied Informatics at the University of Macedonia, in bcs applied informatics.  
It investigates the application of **Q-Learning**, a model-free reinforcement learning algorithm, in **grid-based navigation tasks**.

Multiple agents were implemented and evaluated in discrete grid environments with varying complexity. The study focuses on how different **reward systems (sparse vs dense)**, **reward distributions**, **state-space representations**, and **hyperparameter configurations** affect learning speed, convergence, and path optimality.

The environments and agents were implemented from scratch in Python to ensure full control over the learning process and experimental setup.

You can check out my thesis here https://stergioskokorotsikos.com/assets/Reinforcement%20Learning%20in%20Grid%20Navigation.pdf

### Build with
The project is made purely in python, using the following libraries for plotting, image rendering, info gaining and faster training
* NumPy
* PIL (Python Imaging Library)
* Tkinter (for visualization)

<!-- INSTALLATION AND RUNNING -->
## Installation and Running
To run the model yourself you need the following

* python 3
* install said libraries

You also need to add the path where the csv of the resulting training data and images would be saved to. You can edit the path here
```
# Save csv
df_training.to_csv('FinalResults/Customers/Final/' + mapName + '/csv/' + agentName + '.csv')

display_test(agent, rewards, grids, len(grids), agentName, 'FinalResults/Customers/Final/' + mapName)
```

### Parameters
You are able to change multiple parameters while still achieving a successful training

These parameters are

* The two maps (edit the two maps that are 2d string arrays, where '0' is an empty space, '1' a wall, 'T' the taxi agent and 'G' the goal)
* Reward type (immediate or delayed)
* If punishing technique should be used
* Exploration type used (softmax or greedy)
* Hyperparameters (learning rate, epsilon, discount factor)
* Number of training episodes and display results

There are premade agents on the python code, you are able to comment/uncomment which agents you want to use

```
# agent = TaxiAgent(qTable, reward_type='immediate', exploration='softmax', punished=False, initial_learning_rate=0.01, min_learning_rate=0.001, epsilon_decay=0.999995)
agent = TaxiAgent(qTable, reward_type='immediate', exploration='greedy', punished=False, initial_learning_rate=0.01, min_learning_rate=0.001, epsilon_decay=0.999995)
```

The project also supports visualization of agent movement during evaluation to better understand learned policies.

![til](repo_assets/trained_agent_path_example.gif)

For detailed explanations of configurations and experiments, refer to the thesis document.

## Contact

Stergios Kokorotsikos
University of Macedonia – Department of Applied Informatics
LinkedIn: [Kokorotsikos Stergios](https://www.linkedin.com/in/stergios-kokorotsikos-942248223/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
