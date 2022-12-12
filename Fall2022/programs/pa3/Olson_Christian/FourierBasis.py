import numpy as np
from itertools import product
class FourierBasis:
    def __init__(self, order, d=3):
        # size of feature vector
        self.__n = (order + 1) ** d

        # create all possible combinations of integer coefficents for basis functions
        a = [list(np.linspace(0, order, order + 1, dtype=int))] * d
        self.__c = np.array(list(product(*a)))

    def Q(self, params, state, action):
        # get features and compute q(s,a,w)
        F = self.features(state, action)
        return np.dot(params, F)

    # state and action values should be normalized
    def features(self, state, action):
        # compute features
        x = np.array(state + [action])
        i = np.linspace(0, self.__n - 1, self.__n, dtype=int)
        return np.cos(np.pi * np.dot(self.__c[i], x))

    def getN(self):
        return self.__n