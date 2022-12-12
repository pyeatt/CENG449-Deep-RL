'''
Author: Caleb Ehrisman
Course- Advanced AI CSC-549
Assignment - Programming Assignment #3 - Mountain Car
This file contains the code to implement the SARSA(lambda) algorithm.
All functions needed by solely the agent are included as member functions of class Agent
'''
import numpy as np
import itertools


class FourierBasis:
    def __init__(self, state_space, order):
        self.order = order
        self.state_dim = state_space
        self.order = [order]*self.state_dim
        self.coeff = self.coefficients()

    def coefficients(self):
        """
            FourierBasis.coefficients creates the coeffs for the FourierBasis

            :return np.array(coeff)
        """
        coeff = [np.zeros([self.state_dim])]

        for i in range(0, self.state_dim):
            for c in range(0, self.order[i]):
                v = np.zeros(self.state_dim)
                v[i] = c + 1
                coeff.append(v)
        return np.array(coeff)

    def get_features(self, state):
        """
            FourierBasis.get_features gets the feature vector. Usually noted as x in SARSA(LAMBDA)

            :param state

            :return feature_vector
        """
        return np.cos(np.pi * np.dot(self.coeff, state))

