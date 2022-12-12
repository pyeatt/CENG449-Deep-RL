import gym
import matplotlib
import math
import itertools
import numpy as np
from matplotlib import pyplot as plt
# Bereket Tekeste, AI Program-3

class Fourier:
    def __init__(self, order, size=0):
        self.gradFactors = np.array([])
        self.order = [order] * size
        self.coefs = np.array([i for i in itertools.product(*[range(0, r + 1) for r in self.order])])
        self.gradFactors = 1.0 / np.linalg.norm(self.coefs, ord=2, axis=1)
        self.gradFactors[0] = 1

    def eval(self, state_vector):
        return np.cos(np.pi * np.dot(self.coefs, np.array(state_vector)))


class SarsaLambda:
    def __init__(self, alpha=0.001, gamma=1.0, lamb=0.9, order=3, dim=0):
        self.gamma = gamma
        self.lamb = lamb
        self.alpha = alpha
        self.order = order
        self.fa = []
        for _ in range(0, 3):
            self.fa.append(Fourier(order=order, size=dim))

        self.weights = np.zeros([self.fa[0].coefs.shape[0], 3])
        self.lambWeights = np.zeros(self.weights.shape)

    def nextAction(self, state):
        best = -math.inf
        for act in range(0, 3):
            q = self.Q(state, act)
            if (math.isclose(q, best)):
                bestYet = act
            elif (q > best):
                best = q
                bestYet = act
        return bestYet

    def Q(self, state, action):
        return np.dot(self.weights[:, action], self.fa[action].eval(state))

    def getBestQ(self, state):
        best = -math.inf
        bestYet = -math.inf
        for act in range(0, 3):
            q = self.Q(state, act)
            if (q > best):
                best = q
                bestYet = act
        return best, bestYet

    def update(self, s, a, reward, sn, an=None, isTerminal=False):
        d = reward - self.Q(s, a)
        if not isTerminal:
            if an is not None:
                d += self.gamma * self.Q(sn, an)
            else:
                (qp, an) = self.getBestQ(sn)
                d += self.gamma * self.getBestQ(qp)
        funcEval = self.fa[a].eval(s)

        for act in range(0, 3):
            self.lambWeights[:, act] *= self.gamma * self.lamb
            if act == a:
                self.lambWeights[:, act] += funcEval
            self.weights[:, act] += self.alpha * d * np.multiply(self.fa[act].gradFactors, self.lambWeights[:, act])


if __name__ == "__main__":
    # Parameters
    episodes = 1000
    order = 7
    stepCount = np.zeros([episodes])
    env = gym.make('MountainCar-v0')

    # create SARSA agent
    agent = SarsaLambda(gamma=1.0, lamb=0.9, alpha=0.001, order=order, dim=env.observation_space.shape[0])
    normalized_state = env.observation_space.high - env.observation_space.low

    # Train the agent
    for e in range(0, episodes):
        stepsInEp = 0
        done = False
        agent.lambWeights = np.zeros(agent.weights.shape)
        s = (env.reset()[0] - env.observation_space.low) / normalized_state
        a = agent.nextAction(s)
        while not done:
            sn, r, done, _, data = env.step(a)
            sn = (sn - env.observation_space.low) / normalized_state
            an = agent.nextAction(sn)
            agent.update(s, a, r, sn, an, isTerminal=done)
            s = sn
            a = an
            stepsInEp += 1
        print("Episode number: " + str(e) + " Steps in episode: " + str(stepsInEp))
        stepCount[e] = stepsInEp
    env.close()

    # Make 2D graph
    plt.plot(range(0, episodes), stepCount)
    plt.title("Order: " + str(order))
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.show()

    # Make 3D graph
    plot = plt.figure()
    subPlot = plot.add_subplot(projection='3d')
    x, y = np.meshgrid(np.linspace(0.0, 1.00, 100), np.linspace(0.0, 1.00, 100))
    z = np.zeros(x.shape)

    for i in range(0, z.shape[0]):
        for j in range(0, z.shape[1]):
            state = [x[i, j], y[i, j]]
            (q, _) = agent.getBestQ(state)
            z[i, j] = -1 * q

    subPlot.plot_surface(x, y, z, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
    subPlot.view_init(elev=45, azim=45)
    plt.title('Value Function: Order ' + str(order))
    plt.show()
