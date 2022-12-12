"""
Programming Assignment 1 Gridworld
CSC 449 Advanced Topics in Artificial Intelligence
Christian Olson
09/19/2022

Python 3.9
startup file: main.py

This program calculates an optimal deterministic policy and displays the optimal policy and optimal value function.
The stochastic calculations are included but not used. The policy, reward, value, discount, actions, etc. are stored in
data.py. The all other code for calculating policies is in main.py
"""

import numpy as np
# from data import UP, RIGHT, DOWN, LEFT
from data import threshold
from data import SIZE
from copy import deepcopy
import data


def updateDValue(state, oldValue):
    # calculate value from the next state value under the current deterministic policy using v(s) = r + γv(s')
    current = data.dPolicy[state]
    data.value[state] = data.reward[state][current] + data.discount * oldValue[data.nextState[state][current]]


def updateDPolicy(state):
    # get Maximum value for next states
    maxValue = np.max(data.value[data.nextState[state]])

    # determine if maximum value
    isMax = data.value[data.nextState[state]] == maxValue

    # take first maximum (from boolean actions)
    data.dPolicy[state] = np.argmax(isMax)

"""
def updateSValue(state, oldValue):
    newValue = data.sPolicy[state][UP] * (data.reward[state][UP] + data.discount * oldValue[data.nextState[state][UP]])
    newValue += data.sPolicy[state][RIGHT] * (
                data.reward[state][RIGHT] + data.discount * oldValue[data.nextState[state][RIGHT]])
    newValue += data.sPolicy[state][DOWN] * (
                data.reward[state][DOWN] + data.discount * oldValue[data.nextState[state][DOWN]])
    newValue += data.sPolicy[state][LEFT] * (
                data.reward[state][LEFT] + data.discount * oldValue[data.nextState[state][LEFT]])
    data.value[state] = newValue


def updateSPolicy(state):
    # get Maximum value for next states
    maxValue = np.max(data.value[data.nextState[state]])

    # determine if maximum value
    isMax = data.value[data.nextState[state]] == maxValue

    # set to 1 / number of maximums if maximum or 0 if not
    data.sPolicy[state] = isMax * 1 / np.count_nonzero(isMax)
"""


def learning():
    delta = 1  # initialize to enter loop
    data.value = np.zeros(SIZE)

    # Policy evaluation
    while delta > threshold:
        delta = 0

        # calculate new values
        oldValue = deepcopy(data.value)
        for state in range(SIZE):
            updateDValue(state, oldValue)
            # updateSValue(state, oldValue)

        # maximum of 0 and absolute difference between values of policy iterations
        delta = np.max(np.absolute(data.value - oldValue), delta)

    # Policy improvement
    oldPolicy = deepcopy(data.dPolicy)
    for state in range(SIZE):
        updateDPolicy(state)
        # updateSPolicy(state)

    # returns true if any state policy has changed
    return np.max(oldPolicy != data.dPolicy)


def printResult():
    ACTIONS = ["Up", "Right", "Down", "Left"]

    # create policy string
    string = "Optimal Deterministic Policy π*\n"
    for i in range(4):
        for j in range(4):
            string += ACTIONS[data.dPolicy[i * 4 + j]]
            string += "\t"
        string += "\n"
    print(string)

    # print V*(s)
    print("Optimal Value Funtion\nv*(s) = r(s,π*(s))+" + "{}".format(data.discount) + "v*(s')")


def start():
    while learning():
        pass

    printResult()


if __name__ == '__main__':
    start()
