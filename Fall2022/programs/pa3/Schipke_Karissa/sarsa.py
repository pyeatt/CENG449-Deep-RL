import random
import numpy as np
from mc import MountainCar
import matplotlib.pyplot as plt
from fourier import FourierBasis

class SarsaLam():
    def __init__(self, n, d, feature):
        self.weights = np.zeros((n + 1) ** d)
        self.feature = feature
        self.alpha = 0.001
        self.epsilon = 0.03
        self.gamma = 1
        self.lam = 0.9
        self.mc = MountainCar()
        self.n = n


    def run(self):
        learning_curve = []
        position = np.linspace(-1.2, 0.5, 100)
        velocity = np.linspace(-0.07, 0.07, 100)

        Position, Velocity = np.meshgrid(position, velocity)
        for episode in range(1000):
            self.mc = MountainCar()
            state = self.mc.getState()
            A = self.chooseAction()
            x = self.feature(state + [A]) # x is vector, feature is function doing dot product
            z = np.zeros(len(x))
            Q_old = 0

            num_iterations = 0
            while True:
                num_iterations += 1
                R, p, v = self.mc.takeAction(A)
                A_1 = self.chooseAction()
                x_1 = self.feature([p, v, A_1]) # p, v make state prime
                Q = np.dot(self.weights, x)
                Q_1 = np.dot(self.weights, x_1)
                delta = R + self.gamma * Q_1 - Q
                z = self.gamma * self.lam * z + (1 - self.alpha * self.gamma * self.lam * np.dot(z, x)) * x
                self.weights = self.weights + self.alpha * (delta + Q - Q_old) * z - self.alpha * (Q - Q_old) * x
                Q_old = Q_1
                x = x_1
                A = A_1

                if R == 0:
                    break
            learning_curve.append(num_iterations)

        Z = self.f(position, velocity)
        ax.plot_surface(Position, Velocity, Z, color='white', edgecolors='gray', linewidth=0.25, shade=False)
        
        ax.set_zticks([0, np.round(np.max(Z), 0)])
        # line = np.empty((len(velocity)))
        # line.fill(0.5)
        # ax.plot(line, velocity, ls='-', color='g', linewidth=3)
        if n == 3:
            ax2.plot(np.array(learning_curve), 'b-')
        if n == 5:
            ax2.plot(np.array(learning_curve), 'r-')
        if n == 7:
            ax2.plot(np.array(learning_curve), 'g-')

        return learning_curve

    def f(self, position, velocity):
        z = np.zeros((len(position), len(velocity)))
        for i, x in enumerate(position):
            for j, y in enumerate(velocity):
                self.mc.setState(x, y)
                z[i, j] = -1 * self.calculateValue(self.chooseAction(0))
        return z

    def chooseAction(self, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            return random.choice([-1, 0, 1])
        else:
            best = []
            bestValue = None
            for a in [-1, 0, 1]:
                value = self.calculateValue(a)
                if bestValue is None or bestValue < value:
                    bestValue = value
                    best = [a]
                elif bestValue == value:
                    best.append(a)
            return random.choice(best)

    def calculateValue(self, a):
        return np.dot(self.weights, self.feature(self.mc.getState() + [a]))


if __name__ == '__main__':
    d = 3
    curves = []
    for n in [3, 5, 7]:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('Order ' + str(n) + ' - 1000 Episodes')
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_xlim(-1.2, 0.5)
        ax.set_ylim(-0.07, 0.07)
        ax.set_xticks([-1.2, 0.5])
        ax.set_yticks([-0.07, 0.07])

        fig2 = plt.figure()
        ax2 = plt.axes()
        ax2.set_title('Order ' + str(n) + ' - Learning Curve')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_ylim(0, 1000)

        fb = FourierBasis(n, d)
        sl = SarsaLam(n, d, fb.feature)
        curves.append(sl.run())
        fig.savefig('SurfacePlotOrder' + str(n) + '.pdf')
        fig.savefig('SurfacePlotOrder' + str(n) + '.png')
        fig2.savefig('LearningCurveOrder' + str(n) + '.pdf')
        fig2.savefig('LearningCurveOrder' + str(n) + '.png')

    fig3 = plt.figure()
    ax3 = plt.axes()
    ax3.set_title('All Orders - Learning Curve')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps')
    ax3.set_ylim(0, 1000)
    ax3.plot(np.array(curves[0]), 'b-')
    ax3.plot(np.array(curves[1]), 'r-')
    ax3.plot(np.array(curves[2]), 'g-')
    fig3.savefig('LearningCurveAllOrders.pdf')
    fig3.savefig('LearningCurveAllOrders.png')
