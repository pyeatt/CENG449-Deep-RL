import numpy as np
import math
from Basis import FourierBasis
from Basis import RadialBasis


class SarsaLambda:
    def __init__(self, order=3, dimension=3, basis="Fourier", actions=3, alpha=0.0, gamma=0.0, lamb=0.0, epsilon=0.0):
        self.dimension = dimension
        self.order = order
        self.gamma = gamma
        self.lamb = lamb
        self.epsilon = epsilon
        self.alpha = alpha
        self.actions = actions
        self.actionFunctions = []

        # Set basis type
        if basis == "Fourier":
            for _ in range(0, actions):
                self.actionFunctions.append(FourierBasis(order=order, dim=dimension))
        elif basis == "Radial":
            for _ in range(0, actions):
                self.actionFunctions.append(RadialBasis(order=order, dim=dimension))

        self.weights = np.zeros([self.actionFunctions[0].coefficients.shape[0], actions])
        self.lambdaWeights = np.zeros(self.weights.shape)

    def Q(self, state, action):
        return np.dot(self.weights[:, action], self.actionFunctions[action].evaluate(state))

    def maxQ(self, state):
        best = -math.inf
        for a in range(0, self.actions):
            qval = self.Q(state, a)
            if qval > best:
                best = qval
                bestAction = a
        return best, bestAction

    def wipeTraces(self):
        self.lambdaWeights = np.zeros(self.weights.shape)

    def nextAction(self, state):
        best = -math.inf
        if np.random.random() <= self.epsilon:  # Check if action should be random
            return np.random.randint(low=0, high=self.actions)

        for action in range(0, self.actions):  # Choose best action
            q = self.Q(state, action)
            if math.isclose(q, best):
                bestAction = action
            elif q > best:
                best = q
                bestAction = action

        return bestAction

    def update(self, state, action, reward, splus, aplus=None, terminal=False):
        delta = reward - self.Q(state, action)  # find delta
        if not terminal:
            if aplus is None:
                (qplus, aplus) = self.maxQ(splus)
                (q, _) = self.maxQ(qplus)
                delta += self.gamma * q
            else:
                delta += self.gamma * self.Q(splus, aplus)

        evalB = self.actionFunctions[action].evaluate(state)  # Evaluate for the given action

        for act in range(0, self.actions):
            self.lambdaWeights[:, act] *= self.gamma * self.lamb
            if act == action:
                self.lambdaWeights[:, act] += evalB
            self.weights[:, act] += self.alpha * delta * np.multiply(self.actionFunctions[act].gradient,
                                                                     self.lambdaWeights[:, act])
