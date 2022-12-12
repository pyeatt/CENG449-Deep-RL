import numpy as np
import matplotlib.pyplot as plt
import math
import time

class MCSim():
    # gravity: m/s^2
    # x: m
    # force: N
    # mass: kg
    # timestep: s
    def __init__(self, gravity: float = 9.81, x: float = math.pi, force: float = 40, mass: float = 10, timestep: float = 0.01, show: bool = False):
        self.gravity = gravity
        self.x = x
        self.y = self.hillFunction(x)
        self.force = force
        self.mass = mass
        self.timestep = timestep
        self.xVelocity = 0
        self.lastDirection = None
        self.xGravity = None

        if (show):
            # set up visualization
            fig, ax = plt.subplots()
            self.fig = fig
            self.ax = ax

            lineX = np.arange(-1.2, 0.5, 0.01)
            lineY = np.empty(lineX.shape)
            for i in range(len(lineX)):
                lineY[i] = self.hillFunction(lineX[i])

            line, = self.ax.plot(lineX, lineY)
            self.hillLine = line

            dot, = self.ax.plot(self.x, self.y, marker = 'o', color = 'red')
            self.carDot = dot

            plt.xlim([-1.3, 0.6])
            plt.ion()
            plt.show()

    def stepNoUpdate(self, direction: int):
        if (abs(direction) != 1 and direction != 0):
            raise ValueError("Expected -1, 0, or 1 for direction, got " + str(direction))

        xVelocity = self.xVelocity + 0.001 * direction - 0.0025 * math.cos(3 * self.x)
        x = self.x + xVelocity

        if (x < -1.2):
            x = -1.2
            xVelocity = 0

        if (xVelocity < -0.07):
            xVelocity = -0.07
        elif (xVelocity > 0.07):
            xVelocity = 0.07

        return np.array((x, xVelocity, direction))

    def step(self, direction: int):
        state = self.stepNoUpdate(direction)
        self.x = state[0]
        self.xVelocity = state[1]
        self.y = self.hillFunction(self.x)

        self.lastDirection = direction

        # goal reward
        if (self.x >= 0.5):
            return 0
        # non-goal reward
        else:
            return -1

    def draw(self):
        self.carDot.set_xdata(self.x)
        self.carDot.set_ydata(self.y)

    # represents y displacement for a given x
    def hillFunction(self, x: float) -> float:
        return math.sin(3 * x)

# run a test if this file is being run directly as a script
# car will start in the middle of the valley and attempt to drive up the side,
#  it shouldn't be able to make it all the way up with the default settings.
if __name__ == "__main__":
    sim = MCSim(show = True, timestep = 0.1)
    i = 0
    direction = -1

    while (i < 10):
        print(i)
        sim.step(direction)
        sim.draw()
        i += sim.timestep
        plt.pause(sim.timestep)

        if (sim.x > 4):
            direction = 0
