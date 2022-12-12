import numpy as np

X_STATES = 11
X_VELS = 11
THETA_STATES = 22
THETA_VELS = 11
ACTIONS = 5

STAY = 0
LEFT = 1
RIGHT = 2
LEFT_H = 3
RIGHT_H = 4

PUSH = [0.0, -10.0, 10.0, -50.0, 50.0]

APPLY_FORCE = 0
SET_STATE = 1
NEW_STATE = 2
ANIMATE = 3


def choose_random_demo_start():
    x = np.random.choice((-1.0, 1.0))
    x_dot = np.random.choice((-0.5, 0.5))
    theta = np.random.choice((-.07, .07))
    theta_dot = np.random.choice((-.5, .05))
    return x, x_dot, theta, theta_dot

def choose_random_start():
    x = np.random.choice((-1.0, 1.0))
    x_dot = np.random.choice((-1, 1))
    theta = np.random.choice((-.1, .1))
    theta_dot = np.random.choice((-.5, .5))
    return x, x_dot, theta, theta_dot


def get_action(x, x_dot, theta, theta_dot, epsilon, q_values):
    if np.random.random() < epsilon:  # choose optimal action
        return np.argmax(q_values[:, x, x_dot, theta, theta_dot])
    else:  # choose a random action
        return np.random.randint(ACTIONS)


def bin_state(x, x_dot, theta, theta_dot):
    x = round(x + 5)
    if x < 0:
        x = 0
    if x > 10:
        x = 10

    x_dot = x_dot + 5
    if x_dot < 0:
        x_dot = 0
    if x_dot > 10:
        x_dot = 10
    x_dot = round(x_dot)

    if theta > 0:
        theta = round(abs(theta / (np.pi / 2) * 10))
        if theta > 0:
            theta = theta + 10
    elif theta < 0:
        theta = round(abs(theta / (-np.pi / 2) * 10))
    else:
        theta = 0

    if theta > 21:
        theta = 21

    theta_dot = theta_dot + 5
    if theta_dot < 0:
        theta_dot = 0
    if theta_dot > 10:
        theta_dot = 10
    theta_dot = round(theta_dot)

    return x, x_dot, theta, theta_dot
