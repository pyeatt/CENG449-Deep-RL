from math import cos
from random import random

class MountainCar:

    FORWARD = 1
    BACKWARD = -1
    IDLE = 0

    def __init__(self):
        self.__position = random() - 0.6
        self.__velocity = 0
        self.__finished = False

    def update(self, action):
        # calulate next velocity and position
        vNext = self.__velocity + 0.001 * action - 0.0025 * cos(3 * self.__position)
        vNext = self.__boundV(vNext)
        pNext = self.__position + vNext
        pNext, vNext = self.__boundP(pNext, vNext)

        # update
        self.__velocity = vNext
        self.__position = pNext

        # calculate reward
        if self.isFinished():
            return 0
        return -1

    def __boundV(self, v):
        # bound velocity
        if v > 0.07:
            return 0.07
        elif v < -0.07:
            return -0.07
        return v

    def __boundP(self, p, v):
        # bound position
        if p >= 0.5:
            # reached top
            self.__finished = True
            return 0.5, v
        elif self.__position < -1.2:
            # reset velocity at top of rear hill
            return -1.2, 0
        return p, v

    def reset(self):
        self.__position = random() - 0.6
        self.__velocity = 0
        self.__finished = False

    def getState(self):
        return [self.__position, self.__velocity]

    def setState(self, x, v):
        v = self.__boundV(v)
        self.__position, self.__velocity = self.__boundP(x,v)


    def isFinished(self):
        return self.__finished

    @staticmethod
    def normState(state):
        x = (state[0] + 1.2) / 1.7
        v = (state[1] + 0.07) / 0.14
        return [x, v]

    @staticmethod
    def normAction(action):
        return (action + 1) / 2

    def getNormState(self):
        return MountainCar.normState(self.getState())



