import numpy as np
import itertools
import math


class FourierBasis:
    def __init__(self, order, dim):
        self.coefficients = np.array([])
        self.gradient = np.array([])
        self.order = [order] * dim
        self.coefficients = np.array([itr for itr in itertools.product(*[range(0, x + 1) for x in self.order])])
        with np.errstate(divide='ignore', invalid='ignore'):
            self.gradient = 1.0 / np.linalg.norm(self.coefficients, ord=2, axis=1)
            self.gradient[0] = 1.0

    def evaluate(self, state_vector):
        if type(state_vector) is not np.ndarray:
            state_vector = np.array(state_vector)
        return np.cos(np.pi * np.dot(self.coefficients, state_vector))


class RadialBasis:
    def __init__(self, order, dim):
        self.coefficients = np.array([])
        self.gradient = np.array([])
        self.order = [order] * dim
        self.sigSq = 2.0 / (order - 1.0)
        self.ft = 1.0/(math.sqrt(2.0*math.pi*self.sigSq))
        self.coefficients = np.array([itr for itr in itertools.product(*[range(0, x + 1) for x in self.order])])
        with np.errstate(divide='ignore', invalid='ignore'):
            self.gradient = 1.0 / np.linalg.norm(self.coefficients, ord=2, axis=1)
            self.gradient[0] = 1.0

    def evaluate(self, state_vector):
        if type(state_vector) is not np.ndarray:
            state_vector = np.array(state_vector)
        outV = []
        for ci in self.coefficients:
            cMx = ci-state_vector
            mag = math.sqrt(cMx.dot(cMx))
            magSq = mag * mag
            st = math.exp(-magSq / (2.0 * self.sigSq))
            outV.append(self.ft * st)
        return np.array(outV)
