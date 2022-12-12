import numpy as np
import matplotlib.pyplot as plt
import mountain_car_sim as mcs
import fourier_basis as fb
import random
import time
import os
import math
import utils
import pdb

# simulator parameters
force = 20
mass = 10
timestep = 0.1

# learning parameters
numEpisodes = 1000
gamma = 1 # discount factor
lamb = 0.9 # trace decay factor. python uses lambda as a reserved word >:(
#epsilon = 0.05
epsilon = 0

# fourier basis parameters
numStateVariables = 3 # x position, x velocity, and action
fourierOrder = 3

basis = fb.FourierBasis(numStateVariables, fourierOrder)

# set up an alpha (learning rate) array for each basis function
# higher-order functions with higher-frequency cosine waves will
#  learn more slowly and lower-frequency components will learn
#  more quickly
alpha1 = 0.001
alpha = np.empty(len(basis.cVectors))
alpha[0] = alpha1 # avoid a division by zero
for i in range(1, len(alpha)):
    alpha[i] = alpha1 / np.linalg.norm(basis.cVectors[i], 2)

w = np.zeros(basis.numBasisFunctions) # feature weights
x = np.zeros(basis.numBasisFunctions) # evaluated feature vector, this step
xPrime = np.zeros(basis.numBasisFunctions) # evaluated feature vector, next step

resultsDestinationSubfolder = time.strftime("%Y-%m-%d_%H-%M-%S")
resultsDestination = "results/" + resultsDestinationSubfolder
try:
    os.mkdir("results")
    os.mkdir(resultsDestination)
except FileExistsError:
    os.mkdir(resultsDestination)
except:
    raise

params = {}
params['force'] = force
params['mass'] = mass
params['timestep'] = timestep
params['epsilon'] = epsilon
params['alpha1'] = alpha1
params['order'] = fourierOrder
np.save(resultsDestination + "/params", params)

stepsPerEpisode = []

for episodeIndex in range(numEpisodes):
    sim = mcs.MCSim(force = force, mass = mass, timestep = timestep)
    xPosStart = (random.random() / 5) - 0.6
    sim.x = xPosStart

    A = utils.chooseActionEGreedy(sim, w, basis)

    x = basis.calculate(sim.stepNoUpdate(A))

    z = np.zeros(basis.numBasisFunctions)

    QOld = 0

    numSteps = 0

    while sim.x < 0.5:
        #if (numSteps > 1000):
        #    print(sim.x, sim.xVelocity, A)

        R = sim.step(A)
        APrime = utils.chooseActionEGreedy(sim, w, basis, epsilon)
        xPrime = basis.calculate(sim.stepNoUpdate(APrime))

        Q = w @ x
        QPrime = w @ xPrime

        delta = R + gamma * QPrime - Q
        z = gamma * lamb * z + (1 - alpha * gamma * lamb * (z @ x)) * x
        w1 = (alpha * (delta + Q - QOld) * z)
        w2 = (alpha * (Q - QOld) * x)
        w = w + w1 - w2
        
        QOld = QPrime
        x = xPrime
        A = APrime

        numSteps += 1

    print("Episode " + str(episodeIndex) + " complete, " + str(numSteps) + " steps.")
    stepsPerEpisode.append(numSteps)

    if (episodeIndex % 100 == 0):
        np.save(resultsDestination + "/" + str(episodeIndex) + "-weights", w)
        np.save(resultsDestination + "/" + str(episodeIndex) + "-coeff", basis.cVectors)
        np.save(resultsDestination + "/" + str(episodeIndex) + "-steps", stepsPerEpisode)
        

np.save(resultsDestination + "/" + str(episodeIndex) + "-weights", w)
np.save(resultsDestination + "/" + str(episodeIndex) + "-coeff", basis.cVectors)
np.save(resultsDestination + "/" + str(episodeIndex) + "-steps", stepsPerEpisode)
