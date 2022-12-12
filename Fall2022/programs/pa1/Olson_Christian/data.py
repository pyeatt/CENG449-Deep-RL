import numpy as np

# actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# size of gridworld
SIZE = 16

# acceptable difference for value estimation
threshold = 0.01

discount = 0.99

# Deterministic policy[state]
dPolicy = [LEFT, RIGHT, UP, DOWN,
           DOWN, LEFT, RIGHT, UP,
           RIGHT, DOWN, UP, LEFT,
           RIGHT, RIGHT, UP, DOWN]

# Stochastic policy[state][Up, Right, Down, Left]
sPolicy = [[0.25, 0.25, 0.25, 0.25],  # State 0 random
           [0.25, 0.25, 0.25, 0.25],  # State 1 random
           [0.25, 0.25, 0.25, 0.25],  # State 2 random
           [0.25, 0.25, 0.25, 0.25],  # State 3 random
           [0.25, 0.25, 0.25, 0.25],  # State 4 random
           [0.25, 0.25, 0.25, 0.25],  # State 5 random
           [0.25, 0.25, 0.25, 0.25],  # State 6 random
           [0.25, 0.25, 0.25, 0.25],  # State 7 random
           [0.25, 0.25, 0.25, 0.25],  # State 8 random
           [0.25, 0.25, 0.25, 0.25],  # State 9 random
           [0.25, 0.25, 0.25, 0.25],  # State 10 random
           [0.25, 0.25, 0.25, 0.25],  # State 11 random
           [0.25, 0.25, 0.25, 0.25],  # State 12 random
           [0.25, 0.25, 0.25, 0.25],  # State 13 random
           [0.25, 0.25, 0.25, 0.25],  # State 14 random
           [0, 0, 0, 0]]  # State 15 random final

sPolicy = np.array(sPolicy)

reward = [[-1, -1, -1, -1],  # State 0 random
          [-1, -1, -1, -1],  # State 1 random
          [-1, -1, -1, -1],  # State 2 random
          [-1, -1, -1, -1],  # State 3 random
          [-1, -1, -1, -1],  # State 4 random
          [-1, -1, -1, -1],  # State 5 random
          [-1, -1, -1, -1],  # State 6 random
          [-1, -1, -1, -1],  # State 7 random
          [-1, -1, -1, -2],  # State 8 random Left to 15 = -2
          [-1, -1, -1, -1],  # State 9 random
          [-1, -1, -1, -1],  # State 10 random
          [-1, -1, -1, -1],  # State 11 random
          [-1, -1, -1, -1],  # State 12 random
          [-1, -1, -1, -1],  # State 13 random
          [-1, -1, -1, -1],  # State 14 random
          [-1, 0, 0, -1]]  # State 15 random terminal right and down

reward = np.array(reward)

# the next possible states for each state
nextState = [[0, 1, 4, 0],  # From State 0
             [1, 2, 5, 0],  # From State 1
             [2, 3, 6, 1],  # From State 2
             [3, 3, 7, 2],  # From State 3
             [0, 5, 8, 4],  # From State 4
             [1, 6, 9, 4],  # From State 5
             [2, 7, 10, 5],  # From State 6
             [3, 7, 11, 6],  # From State 7
             [4, 9, 12, 15],  # From State 8 Left to 15
             [5, 10, 13, 8],  # From State 9
             [6, 11, 14, 9],  # From State 10
             [7, 11, 15, 10],  # From State 11
             [8, 13, 12, 12],  # From State 12
             [9, 14, 13, 12],  # From State 13
             [10, 15, 14, 13],  # From State 14
             [11, 15, 15, 14]]  # From State 15

nextState = np.array(nextState)

# initial values are 0
value = np.zeros(SIZE)
