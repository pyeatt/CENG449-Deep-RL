# Karissa Schipke
# Python 3.10.7

#!/usr/bin/env python3

from scipy.integrate import solve_ivp
import inverted_pendulum as ip
import json
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import struct
import random
import zmq

GAMMA = 0.99
APPLY_FORCE = 0
SET_STATE = 1
NEW_STATE = 2
ANIMATE = 3

class SocketHandler():

    def __init__(self):
        self.__animate = True
        self.__context = zmq.Context()
        self.__socket = self.__context.socket(zmq.REQ)
        self.__socket.connect("tcp://localhost:5555")

    def SetState(self, state):
        command = SET_STATE
        x, xdot, theta, thetadot = state
        request_bytes = struct.pack(
            'iffff', command, x, xdot, theta, thetadot)
        self.__socket.send(request_bytes)
        S0, R = self.Receive()
        if abs(S0[0]) > 5:
            R = -1
        if abs(S0[2]) > np.pi/4:
            R = -1
        return S0, R

    def ToggleAnimation(self):
        # toggle animation
        command = ANIMATE
        request_bytes = struct.pack('ii', command, self.__animate)
        self.__socket.send(request_bytes)
        self.__animate = not self.__animate
        self.Receive()

    def TakeAction(self, action):
        command = APPLY_FORCE
        request_bytes = struct.pack('if', command, action)
        self.__socket.send(request_bytes)
        return self.Receive()

    def Receive(self):
        response_bytes = self.__socket.recv()
        response_command, = struct.unpack('i', response_bytes[0:4])

        if response_command == NEW_STATE:
            x, xdot, theta, thetadot, reward = struct.unpack(
                'fffff', response_bytes[4:])
            new_state = (x, xdot, theta, thetadot)
            return new_state, reward
        elif response_command == ANIMATE:
            animation_enabled, = struct.unpack('i', response_bytes[4:])
        else:
            print("Error: invalid command: ", response_command)

        return None

def Discretize(S):
    # 11 states, 3 states (-1, 0, 1), 160+ states
    S_new = (round(S[0]), np.sign(S[1]), round(S[2], 2), np.sign(S[3]))
    return S_new

def EpsilonGreedy(Q, S, epsilon):
    actions = [-1000, 1]
    best = None
    if random.random() < epsilon:
        return random.choice(actions)

    for a in actions:
        if best is None or Q.get((S, a), 0) > best:
            best = Q.get((S, a), 0)
            bestAction = a
    return bestAction

# Algorithm parameters: step size alpha is in (0, 1], small epsilon > 0
def Sarsa(alpha, epsilon):
    # Initialize Q(s, a), for all s in S+, a in A(s), arbitrarily except that Q(terminal, ·)=0
    Q = {}
    
    socketHandler = SocketHandler()
    socketHandler.ToggleAnimation()
    
    # Loop for each episode:
    while True:
        # Initialize S
        x = random.randrange(-4, 4)
        xdot = random.randrange(-10, 10)
        theta = random.choice(np.linspace(-np.pi/6, np.pi/6, 25))
        thetadot = random.randrange(-10, 10)
        S, R = socketHandler.SetState([x, xdot, theta, thetadot])
        # Choose A from S using policy derived from Q (e.g., epsilon-greedy)
        A = EpsilonGreedy(Q, S, epsilon)
        # Loop for each step of episode:
        for _ in range(2500):
            # Take action A, observe R, S0
            S0, R = socketHandler.TakeAction(A)
            # Choose A0 from S0 using policy derived from Q (e.g., epsilon-greedy)
            A0 = EpsilonGreedy(Q, S0, epsilon)

            S = Discretize(S)
            S0 = Discretize(S0)
        
            # Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S0, A0) - Q(S, A)]
            Q[S, A] = Q.get((S, A), 0) + alpha * (R + GAMMA * Q.get((S0, A0), 0) - Q.get((S, A), 0)) 
            # S <- S0
            S = S0
            # A <- A0
            A = A0
            if R < 0:
                break

if __name__ == "__main__":
    Sarsa(0.4, 0.9)
