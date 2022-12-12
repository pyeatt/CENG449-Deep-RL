# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 20:28:31 2022

@author: Colton Snyder
"""

from enum import Enum
import math
import numpy as np
import pickle
import random
import sys

class Actions(Enum):
    LEFT = -1
    RIGHT = 1
    IDLE = 0
    
    
    
# Define Hyperparameters - Using parameters from Konidaris et al. paper
STEP_SIZE = 0.001
LAMBDA = 0.9
EPSILON = 0.01
GAMMA = 1.0


fourierBasisOrder = 3
numBasisFuncs = (fourierBasisOrder + 1) ** 2 + 1


def updateState(X, XDot, force):
    newXDot = XDot + 0.001 * force - 0.0025 * math.cos(3*X)
    newXDot = max(-0.07, min(0.07, newXDot))
    
    newX = X + newXDot
    newX = max(-1.2, newX)
    
    return newX, newXDot


def determineFourierBasis(X, XDot, action):
    basis = []
    for i in range(fourierBasisOrder + 1):
        for j in range(fourierBasisOrder + 1):
            basis.append(math.cos(math.pi * (i * X + j * XDot)))
    basis.append(action)
    return basis


# Choose next action by e-Greedy policy
def nextAction(X, XDot, weights, EPSILON):
    if random.random() > EPSILON:
        # basis = determineFourierBasis(X, XDot)
        # basis.append(0)
        
        max_val = -math.inf
        max_act = Actions.LEFT
        for act in Actions:
            # basis[-1] = act.value
            basis = determineFourierBasis(X, XDot, act.value)
            val = np.dot(weights, basis)
            if (val > max_val):
                max_val = val
                max_act = act
        return max_act
    else:
        act_ind = random.randint(0, len(list(Actions))-1)
        return list(Actions)[act_ind]


def main(fileName = None, numEpisodes=20):
    episodeNum = 0
    episodeRewards = []
    
    weights = np.random.random((numBasisFuncs))#np.zeros((numBasisFuncs))# + 1))
    
    while episodeNum < numEpisodes:
        episodeNum += 1
        # Initialize state
        X = random.random() * 0.2 - 0.6 # Random starting position in [-0.6, -0.4)
        XDot = 0 # Zero Initial Velocity
        totalReward = 0
        
        # Choose starting Action
        A = nextAction(X, XDot, weights, EPSILON)
        A = A.value
        features = determineFourierBasis(X, XDot, A)
        z = np.zeros((numBasisFuncs))
        Qold = 0
        
        while X < 0.5:
            X, XDot = updateState(X, XDot, A)
            reward = -1
            if X >= 0.5:
                reward = 0
            totalReward += reward
            
            
            APrime = nextAction(X, XDot, weights, EPSILON)
            APrime = APrime.value
            featuresPrime = determineFourierBasis(X, XDot, APrime)# + [APrime]
            
            Q = np.dot(weights, features)
            QPrime = np.dot(weights, featuresPrime)
            
            # if totalReward % 1000 == 0:
            #     print(f"{Q=}, {X=}, {XDot=}, {A=}")
            
            delta = reward + GAMMA * QPrime - Q
            z = GAMMA * LAMBDA * z + (1 - STEP_SIZE*GAMMA*LAMBDA*np.dot(z, features)) * np.asarray(features)
            weights = weights + STEP_SIZE * (delta + Q - Qold) * z - STEP_SIZE * (Q - Qold) * np.asarray(features)
            
            Qold = QPrime
            features = featuresPrime
            A = APrime
        
        print(f"Finished Episode {episodeNum}! {totalReward=}")
        episodeRewards.append(totalReward)
        if fileName:
            with open(fileName, "a") as file:
                file.write("\n")
                file.write(str(totalReward))
    # with open("Order3_weights.txt", "w") as file:
    #     weights.tofile(file)
        
        
            
if __name__ == "__main__":
    file = None
    numEpisodes = 20
    if len(sys.argv) == 2:
        numEpisodes = int(sys.argv[1])
    elif len(sys.argv) == 3:
        file = sys.argv[2]
        numEpisodes = int(sys.argv[1])
    else:
        print("""Usage: python3 pa3.py [<episodes> [<fileName>]]
where episodes is number of episodes to run,
and filename is the name of the file to save episode rewards to.""")
        sys.exit()
    main(file, numEpisodes)