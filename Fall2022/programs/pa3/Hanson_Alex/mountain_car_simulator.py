"""
Author: Alex Hanson
Date: 11/26/2022

Simulation of Mountain car

State variables
Velocity = (-0.07, 0.07)
Position = (-1.2, 0.6)

Actions
motor = (left, neutral, right)

Reward
For each time step:
reward = -1

Update function
For each time step:
Action = [-1, 0, 1]
Velocity = Velocity + (Action) * 0.001 + cos(3 * Position) * (-0.0025)
Position = Position + Velocity

Starting condition
Position = -0.5
Velocity = 0.0

Termination condition
Simulation ends when:
Position >= 0.6
"""

import math # cos
import numpy as np


class MountainCarSimulator:
    MAX_TIME_STEPS = 1000
    MAX_VELOCITY = 0.07
    MIN_VELOCITY = -0.07
    MIN_POSITION = -1.2
    MAX_POSITION = 0.6

    INIT_VELOCITY = 0.0
    INIT_POS = -0.5

    def __init__(self):
        self.initilize()


    def initilize(self) -> None:
        self.velocity = self.INIT_VELOCITY
        self.position = self.INIT_POS
        self.terminal = False
        return self.getState()


    def step(self, action: int):
        r = -1
        # update velocity
        # Velocity = Velocity + (Action) * 0.001 + cos(3 * Position) * (-0.0025)
        self.velocity += (action) * 0.001 + math.cos(3 * self.position) * (-0.0025)
        if self.velocity > self.MAX_VELOCITY:
            self.velocity = self.MAX_VELOCITY
        elif self.velocity < self.MIN_VELOCITY:
            self.velocity = self.MIN_VELOCITY
        # update position
        # Position = Position + Velocity
        self.position += self.velocity
        if self.position <= self.MIN_POSITION:
            self.position = self.MIN_POSITION
            r -= 10 # went over bounds give big negative
        elif self.position >= self.MAX_POSITION:
            self.terminal = True
            r += 1 # terminal give zero
        return np.array([self.position, self.velocity]), r

    def getState(self):
        return np.array([self.position, self.velocity])

    def getReward(self) -> int:
        if self.position >= self.MAX_POSITION:
            return 0
        else:
            return -1

    def isTerminal(self) -> bool:
        return self.terminal







if __name__ == "__main__":
    print("Test Mountain Car Simulator")

    from time import time

    s = time()
    mc = MountainCarSimulator()
    e = time()
    print(f"constructor took {e-s}")

    s = time()
    for i in range(1000):
        mc.step(1)
    e = time()
    print(f"step took {e-s}")


