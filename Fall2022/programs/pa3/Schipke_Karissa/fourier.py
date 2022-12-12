import numpy as np
import itertools as it

class FourierBasis():

    def __init__(self, n, d):
        self.n = n
        self.d = d
        self.c = np.array(list(it.product(range(self.n + 1), repeat = self.d)))

    def feature(self, state):
        state[0] = (state[0] + 1.2) / 1.7
        state[1] = (state[1] + 0.07) / 0.14
        state[2] = (state[2] + 1) / 2
        return np.cos(np.pi * np.dot(self.c, state))
