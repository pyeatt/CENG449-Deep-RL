#!/usr/bin/env python3
import numpy as np
import struct
import zmq
import my_agent_data as data

"""
Programming Assignment 2: Temporal Differencing
Christian Olson

This file starts a trained agent using Jonathan Mathew's simulator. The settings for the agent can be found
in my_agent_data.py.
usage: $python trained_agent.py
       $python inverted_pendulum_server.py <--animate>
           -Requires the simulator to be running along side the agent
"""


def main():
    # connect
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    # read values
    data.Q = np.load("trained.npy")


    while True:
        # start
        initSim(socket)
        state = [getXState(), getXdotState(), getThetaState(), getThetadotState()]
        action = getAction(state)

        #until boundary
        while abs(data.x) != 5:
            reward, nextState, action = doAction(socket, action)

def initSim(socket):
    #start centered and straight
    msg = struct.pack('iffff', data.SET_STATE, 0.0, 0.0, 0.0, 0.0)
    socket.send(msg)
    msg = socket.recv()
    data.x, data.xdot, data.theta, data.thetadot, reward = struct.unpack('fffff', msg[4:])

    # force reaction from agent
    msg = struct.pack('if', data.APPLY_FORCE, 1.0)
    socket.send(msg)
    socket.recv()


def getThetaState():
    # create state ranges
    bounds = np.linspace(data.THETADOT_MIN, data.THETADOT_MAX, data.NUM_THETA_STATES + 1)

    # find state
    for i in range(0, data.NUM_THETA_STATES):
        if data.theta >= bounds[i] and data.theta < bounds[i + 1]:
            return i


def getThetadotState():
    # outside defined range (first or last)
    if data.thetadot < data.THETADOT_MIN:
        return 0
    if data.thetadot >= data.THETADOT_MAX:
        return data.NUM_THETADOT_STATES - 1

    # create state ranges
    bounds = np.linspace(data.THETADOT_MIN, data.THETADOT_MAX, data.NUM_THETADOT_STATES - 1)

    # find state
    for i in range(0, data.NUM_THETADOT_STATES - 2):
        if data.thetadot >= bounds[i] and data.thetadot < bounds[i + 1]:
            return i + 1


def getXdotState():
    # outside defined range (first or last)
    if data.xdot < data.XDOT_MIN:
        return 0
    if data.xdot >= data.XDOT_MAX:
        return data.NUM_XDOT_STATES - 1

    # create state ranges
    bounds = np.linspace(data.XDOT_MIN, data.XDOT_MAX, data.NUM_XDOT_STATES - 1)

    # find state
    for i in range(0, data.NUM_XDOT_STATES - 2):
        if data.xdot >= bounds[i] and data.xdot < bounds[i + 1]:
            return i + 1


def getXState():
    # outside of boundaries
    if data.x == data.X_MIN:
        return 0
    if data.x == data.X_MAX:
        return data.NUM_X_STATES - 1

    # create state boundaries
    bounds = np.linspace(data.X_MIN, data.X_MAX, data.NUM_X_STATES - 1)
    for i in range(0, data.NUM_X_STATES - 2):
        if data.x >= bounds[i] and data.x < bounds[i + 1]:
            return i + 1


def getAction(state):
    # e-greedy if not epsilon != 0
    p = np.random.random()
    if p < data.epsilon:
        # random action
        return np.random.randint(0, data.NUM_ACTIONS)
    else:
        # state from index to maximum value
        return np.argmax(data.Q[state[0], state[1], state[2], state[3]])


def doAction(socket, action):
    # Do action
    msg = struct.pack('if', data.APPLY_FORCE, getForce(action))
    socket.send(msg)

    # get update
    msg = socket.recv()
    data.x, data.xdot, data.theta, data.thetadot, reward = struct.unpack('fffff', msg[4:])

    # return update state data
    nextState = [getXState(), getXdotState(), getThetaState(), getThetadotState()]
    nextAction = getAction(nextState)
    return reward, nextState, nextAction


def getForce(action):
    # force from action state
    if action == data.SMALL_POS:
        return data.SMALL_FORCE
    elif action == data.MED_POS:
        return data.MED_FORCE
    elif action == data.BIG_POS:
        return data.BIG_FORCE
    elif action == data.SMALL_NEG:
        return -data.SMALL_FORCE
    elif action == data.MED_NEG:
        return -data.MED_FORCE
    elif action == data.BIG_NEG:
        return -data.BIG_FORCE


if __name__ == "__main__":
    main()