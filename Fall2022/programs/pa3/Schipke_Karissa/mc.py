import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class MountainCar:
    def __init__(self):
        self.position = random.uniform(-0.6, -0.4)
        self.velocity = 0

    def getState(self):
        return [self.position, self.velocity]

    def setState(self, position, velocity):
        self.position = position
        self.velocity = velocity
        
    def takeAction(self, action):
        v = np.clip(self.velocity + (0.001 * action) - (0.0025 * np.cos(3 * self.position)), -0.07, 0.07)
        p = np.clip(self.position + v, -1.2, 0.5)

        if p == 0.5:
            reward = 0
        else:
            reward = -1

        if p <= -1.2:
            v = max(0, v)

        self.position = p
        self.velocity = v

        return reward, self.position, self.velocity
         
    def animate(self, _):
        plt.clf()
        x = np.linspace(-1.5, 1, 100)
        y = -np.cos(x)
        r, _ , _ = self.takeAction(1)
        print(r)
        plt.plot(x, y)
        plt.plot(self.position, -np.cos(self.position), 'ro')

        if self.position == 0.5:
            plt.close()

if __name__ == '__main__':
    mc = MountainCar()
    fig = plt.figure()
    anim = FuncAnimation(fig, mc.animate, interval = 10)
    plt.show()
    plt.close()