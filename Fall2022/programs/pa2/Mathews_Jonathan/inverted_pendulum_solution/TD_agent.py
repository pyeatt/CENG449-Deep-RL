#!/usr/bin/env python3

from scipy.integrate import solve_ivp
import argparse
import inverted_pendulum as ip
import json
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import struct
import zmq


min_x = 0
max_x = 0
min_xdot = 0
max_xdot = 0
min_theta = 0
max_theta = 0
min_thetadot = 0
max_thetadot = 0


class State():

    # Define how wide the bins are for each state variable
    x_interval = 10
    xdot_interval = 4
    theta_interval = 0.05
    thetadot_interval = 0.5

    def __init__(self, x, xdot, theta, thetadot):
        global min_x
        global max_x
        global min_xdot
        global max_xdot
        global min_theta
        global max_theta
        global min_thetadot
        global max_thetadot

        # determine the bin for each state variable
        self.x = int(np.floor(x/self.x_interval+0.5))
        self.xdot = int(np.floor(xdot/self.xdot_interval+0.5))
        self.theta = int(np.floor(np.sin(theta)/self.theta_interval+0.5))
        self.thetadot = int(np.floor(thetadot/self.thetadot_interval+0.5))

        if self.x < min_x:
            min_x = self.x
        if self.xdot < min_xdot:
            min_xdot = self.xdot
        if self.theta < min_theta:
            min_theta = self.theta
        if self.thetadot < min_thetadot:
            min_thetadot = self.thetadot
        if self.x > max_x:
            max_x = self.x
        if self.xdot > max_xdot:
            max_xdot = self.xdot
        if self.theta > max_theta:
            max_theta = self.theta
        if self.thetadot > max_thetadot:
            max_thetadot = self.thetadot

    def __hash__(self):
        return hash((self.x, self.xdot, self.theta, self.thetadot))

    def __eq__(self, other):

        if isinstance(other, State):
            return (self.x == other.x
                    and self.xdot == other.xdot
                    and self.theta == other.theta
                    and self.thetadot == other.thetadot)
        else:
            return NotImplemented

    def __str__(self):
        return "({},{},{},{})".format(
                self.x*self.x_interval, 
                self.xdot*self.xdot_interval, 
                self.theta*self.theta_interval, 
                self.thetadot*self.thetadot_interval)

    def __repr__(self):
        return str(self)


class Action():
    MIN_FORCE = -10.0
    MAX_FORCE = 10.0
    NUM_BINS = 20
    FORCE_INTERVAL = (MAX_FORCE-MIN_FORCE)/(NUM_BINS-1)

    def __init__(self, index):
        bounded = max(min(force, self.MAX_FORCE), self.MIN_FORCE)
        self.force = int(np.floor(bounded/self.force_interval))

    def __hash__(self):
        return hash(self.force)

    def __eq__(self, other):
        if isinstance(other, Action):
            return self.force == other.force
        else:
            return NotImplemented


def epsilon_greedy(Q, state, epsilon=0.1):
    r = np.random.rand()
    try:
        actions = Q[state]
    except KeyError:
        actions = 0*np.ones(Action.NUM_BINS)
        Q[state] = actions
        print("state:", state)

    if r > epsilon:
        action = np.argmax(actions)

    else:
        action = np.random.randint(0, Action.NUM_BINS)

    return action


def to_force(action):
    force = Action.MIN_FORCE + action*Action.FORCE_INTERVAL
    return force


def animate(socket, animation_enabled):
    APPLY_FORCE = 0
    SET_STATE = 1
    NEW_STATE = 2
    ANIMATE = 3
    command = ANIMATE
    request_bytes = struct.pack('ii', command, animation_enabled)
    socket.send(request_bytes)

    response_bytes = socket.recv()
    response_command, animation_enabled = struct.unpack('ii', response_bytes)
    return animation_enabled


def set_state(socket, x, xdot, theta, thetadot):
    APPLY_FORCE = 0
    SET_STATE = 1
    NEW_STATE = 2
    ANIMATE = 3

    command = SET_STATE
    request_bytes = struct.pack('iffff', command, x, xdot, theta, thetadot)
    socket.send(request_bytes)
    response_bytes = socket.recv()
    response_command, x, xdot, theta, thetadot, reward = struct.unpack(
        'ifffff', response_bytes)


def main(args):
    APPLY_FORCE = 0
    SET_STATE = 1
    NEW_STATE = 2
    ANIMATE = 3

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    global alpha
    global gamma
    global epsilon

    alpha = 0.1
    gamma = 0.95
    epsilon = 0.00

    # Q is stored as a hash-table (dictionary) so we only store the states we have seen
    Q = {}
    state_counter = {}

    # load the save file
    if args.qfile is not None:
        with open(args.qfile, 'r') as f:
            json_dict = json.load(f)

        best_list = []
        for key in json_dict:
            x, xdot, theta, thetadot = eval(key)
            item_dict = json_dict[key]
            item_state = State(x, xdot, theta, thetadot)
            Q[item_state] = item_dict["actions"]
            state_counter[item_state] = item_dict["count"]

    deltas = []
    fig, ax = plt.subplots()

    animation_enabled = False
    best_timestep = 1
    count = 0
    while True:

        # Animate an episode every 1000 episodes
        if count % 1000 == 0:
            animation_enabled = animate(socket, True)

        x = 0
        xdot = 0
        theta = np.random.uniform(-0.02, 0.02)
        thetadot = 0

        state = State(x, xdot, theta, thetadot)
        set_state(socket, x, xdot, theta, thetadot)

        timestep = 0
        episode_finished = False
        while not episode_finished:
            # set x to wrap around so the cart can travel forever in any direction
            if x < -4:
                x += 8
                state = State(x, xdot, theta, thetadot)
                set_state(socket, x, xdot, theta, thetadot)
            if x > 4:
                x -= 8
                state = State(x, xdot, theta, thetadot)
                set_state(socket, x, xdot, theta, thetadot)

            # get a force to apply
            action = epsilon_greedy(Q, state, epsilon)
            u = to_force(action)

            # apply the force
            command = APPLY_FORCE
            request_bytes = struct.pack('if', command, u)
            socket.send(request_bytes)
            response_bytes = socket.recv()
            response_command, new_x, new_xdot, new_theta, new_thetadot, reward = struct.unpack(
                'ifffff', response_bytes)

            # override the simulator's reward function with our own
            if new_theta < -np.pi/4 or new_theta > np.pi/4:
                reward = -1
                episode_finished = True
            else:
                reward = 0

            new_state = State(new_x, new_xdot, new_theta, new_thetadot)

            # punish for going fast
            reward += -0.1*np.abs(new_state.xdot)

            # Apply the Q-learning update step
            QS = Q[state]
            current_best_action = np.argmax(QS)
            QSA = QS[action]
            try:
                QSp = Q[new_state]
            except KeyError:  # If the new state isn't in our dictionary, add it
                QSp = 0 * np.ones(Action.NUM_BINS)
                Q[new_state] = QSp
                print("new_state:", new_state)
                # print the current bounds of our state space
                print("({},{}), ({},{}), ({},{}), ({},{})".format(
                    min_x, max_x, min_xdot, max_xdot, min_theta, max_theta, min_thetadot, max_thetadot))

            delta = alpha*(reward + gamma*np.max(QSp) - QSA)
            deltas.append(delta)
            Q[state][action] = QSA + delta
            new_best_action = np.argmax(Q[state][action])

            # Record how many times a state has been visited
            try:
                state_counter[state] += 1
            except KeyError:
                state_counter[state] = 1

            # Make the new state our current state
            state = new_state
            x = new_x
            xdot = new_xdot
            theta = new_theta
            thetadot = new_thetadot

            # Turn on animation if it's lasted twice as long as before.
            if animation_enabled == False and timestep > best_timestep:
                best_timestep *= 2
                print("best timestep:", best_timestep)
                animation_enabled = animate(socket, True)

            timestep += 1

        print("count", count)

        # Every time we animate an episode, plot additional info and make a save file
        if animation_enabled:
            # plot how much Q is changing over time
            ax.clear()
            ax.plot(deltas[::max(1, int(len(deltas)/5000))], '.')
            plt.draw()
            plt.pause(0.001)
            plt.show(block=False)

            # write a save file
            print("writing")
            temp_dict = {}
            for key in Q:
                try:
                    state_count = state_counter[key]
                except KeyError:
                    state_count = 0
                    state_counter[key] = state_count

                temp_dict[str(key)] = {"actions": list(
                    Q[key]), "count": state_count}

            with open("Q_{:08d}.json".format(count), 'w') as f:
                json_object = json.dumps(temp_dict, indent=2)
                f.write(json_object)
            print("written")

        # Disable animation after every episode
        animation_enabled = animate(socket, False)

        count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qfile")
    args = parser.parse_args()

    main(args)
