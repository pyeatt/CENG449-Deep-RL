import numpy as np
import math
import itertools
import mountain_car_sim as mcs
import pdb

class FourierBasis():
    # provide either
    #  numStateVariables and order
    #  OR
    #  premade cVectors
    def __init__(self, numStateVariables: int = None, order: int = None, cVectors: np.ndarray = None):
        if (numStateVariables != None and order != None):
            if (type(cVectors) != type(None)):
                raise ValueError("Expected (numStateVariables and order) OR (cVectors), not both.")

            self.numStateVariables = numStateVariables
            self.order = order
            self.numBasisFunctions = (self.order + 1) ** self.numStateVariables

            # fill up the c vectors with [0,0], [0,1], [1,0], etc. one for each basis function
            #  so we get our different frequency combinations
            cVectorValues = np.arange(self.order + 1)
            self.cVectors = np.array(list(itertools.product(cVectorValues, repeat = self.numStateVariables)))
        elif (type(cVectors) != type(None)):
            if (numStateVariables != None or order != None):
                raise ValueError("Expected (numStateVariables and order) OR (cVectors), not both.")

            self.cVectors = cVectors
            self.numBasisFunctions = len(cVectors)
        else:
            raise RuntimeError("You're using it wrong :(")

    def calculate(self, state: np.ndarray):
        result = np.empty(self.numBasisFunctions)
        stateArray = None

        stateArray = state.copy()

        # transform position, velocity, and action to be in range [0,1]
        stateArray[0] = (stateArray[0] + 1.2) / (0.5 + 1.2)
        stateArray[1] = (stateArray[1] + 0.07) / (0.07 * 2)
        stateArray[2] = (stateArray[2] + 1) / 2

        for i in range(len(result)):
            result[i] = math.cos(math.pi * (self.cVectors[i] @ stateArray))

        return result
