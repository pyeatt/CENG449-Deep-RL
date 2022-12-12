import numpy as np
# Commands
APPLY_FORCE = 0
SET_STATE = 1
NEW_STATE = 2
ANIMATE = 3

# Actions
#NOTHING = 0
SMALL_POS = 0
MED_POS = 1
BIG_POS = 2
SMALL_NEG = 3
MED_NEG = 4
BIG_NEG = 5
NUM_ACTIONS = 6

# States
NUM_X_STATES = 8
NUM_XDOT_STATES = 4
NUM_THETA_STATES = 8
NUM_THETADOT_STATES = 4

X_MAX = 5
X_MIN = -X_MAX
XDOT_MAX = 1.5
XDOT_MIN = -XDOT_MAX
THETA_MAX = np.pi
THETA_MIN = -THETA_MAX
THETADOT_MAX = np.pi
THETADOT_MIN = -THETADOT_MAX

#values
Q = np.zeros((NUM_X_STATES, NUM_XDOT_STATES, NUM_THETA_STATES, NUM_THETADOT_STATES, NUM_ACTIONS))

#params
stable = False
#NO_FORCE = 0
SMALL_FORCE = 10
MED_FORCE = 50
BIG_FORCE = 100
epsilon = 0.0
stepsize = 0.1
discount = 0.9

# state values
x = 0.0
xdot = 0.0
theta = 0.0
thetadot = 0.0
force = 0.0
