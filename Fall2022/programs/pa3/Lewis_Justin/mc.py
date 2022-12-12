# Mountain car simulator
import math
import numpy as np
import random
import time
from matplotlib import pyplot as plt
from matplotlib import animation

class Car:
    def __init__(self):
        # In accordance to MC problem as defined in the RL book on page 245
        # (example 10.1)
        self.x_bound = [-1.2, 0.5]
        self.x_dot_bound = [-0.07, 0.07]
        self.start_position_range = [-0.6, -0.4]

    def reset(self):
        self.x = random.uniform(self.start_position_range[0],
                self.start_position_range[1])
        # self.x = -1.2
        self.v = 0
        return (self.x, self.v)

    # force \in {-1, 0, 1}
    def apply_force(self, force):
        vp = np.clip(self.v + 0.001*force - 0.0025 * np.cos(3 * self.x),
                                        self.x_dot_bound[0], self.x_dot_bound[1])
        xp = np.clip(self.x + vp, self.x_bound[0], self.x_bound[1])

        # vp *= .98
        if xp == self.x_bound[0]:
            vp = max(0, vp)
        self.x = xp
        self.v = vp

        # return reward, x, v
        return -1 if self.x != self.x_bound[1] else 0, (self.x, self.v)

    def animate(self, i):
        # Added this in to test the mc problem
        # the agent was finishing by just applying rightward force, which had me concerned that 
        # I did the simulation wrong, however upon watching it I can see that the car is going back and
        # forth just in a very unoptimal fashion, so there's certainly room for improvement. 
        r, (x, v) = self.apply_force(1)
        plt.clf()
        plt.xlim((-1.5, 1))
        plt.ylim((.8, 1.2))
        plt.axvline(x=self.x_bound[0], color='b')
        plt.axvline(x=self.x_bound[1], color='b')
        plt.axvline(x=self.x_dot_bound[0], color='g')
        plt.axvline(x=self.x_dot_bound[1], color='g')
        plt.plot([x, v], [1, 1.1], 'ro')
    def animate2(self, i):
        plt.clf()
        x = np.linspace(-1.5, .7, 100)
        y = -np.cos(x)
        self.apply_force(np.sign(self.v))
        plt.plot(x, y)
        plt.plot(self.x, -np.cos(self.x) + 0.1, 'ro')

    

if __name__ == "__main__":
    fig = plt.figure()
    car = Car()
    anim = animation.FuncAnimation(fig, car.animate, init_func=car.reset, interval=10)
    plt.show()
