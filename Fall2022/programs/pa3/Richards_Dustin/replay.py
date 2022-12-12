import sys
import numpy as np
import utils
import fourier_basis as fb
import mountain_car_sim as mcs
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import pdb

def printUsage():
    print("Usage: " + sys.argv[0] + " results_dir episode")

if (len(sys.argv) < 3):
    printUsage()

resultsDir = sys.argv[1] + '/'
episode = sys.argv[2]

weightsFile = resultsDir + episode + "-weights.npy"
coeffsFile = resultsDir + episode + "-coeff.npy"
stepsFile = resultsDir + episode + "-steps.npy"
paramsFile = resultsDir + "params.npy"

w = np.load(weightsFile)
cVectors = np.load(coeffsFile)
stepsPerEpisode = np.load(stepsFile)
params = np.load(paramsFile, allow_pickle = True).item()

force = params['force']
mass = params['mass']
timestep = params['timestep']
order = params['order']

basis = fb.FourierBasis(cVectors = cVectors)

sim = mcs.MCSim(force = force, mass = mass, show = True, timestep = 0.1)
sim.x = -0.5

numSteps = 0

while (sim.x <= 0.5 and numSteps < 10000):
    action = utils.chooseBestAction(sim, w, basis)
    sim.step(action)
    sim.draw()
    plt.pause(0.000000000001)

    numSteps += 1

graphCountPos = 100
graphCountVel = 100

posAxis = np.linspace(-1.2, 0.5, graphCountPos)
velAxis = np.linspace(-0.07, 0.07, graphCountVel)

xx, yy = np.meshgrid(posAxis, velAxis)

valueMatrix = np.empty(xx.shape)

print("Calculating value function graph...")

for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        sim = mcs.MCSim()
        sim.x = xx[i,j]
        sim.xVelocity = yy[i,j]
        bestAction = utils.chooseBestAction(sim, w, basis)
        valueMatrix[i,j] = -utils.getQuality(sim, w, basis, bestAction)

    if ((i+1) % 10 == 0):
        print(str(i+1) + "/" + str(xx.shape[0]))

plt.ioff()

subplotRows = 1
subplotCols = 1

fig = plt.figure()
ax = fig.add_subplot(subplotRows, subplotCols, 1, projection = '3d')
surf = ax.plot_surface(xx, yy, valueMatrix, cmap = cm.coolwarm)
ax.set_xlabel('Position')
ax.set_ylabel('Velocity')
ax.set_title('Fourier Order ' + str(order) + ' Value Function')
fig.colorbar(surf, shrink = 0.5, aspect = 5)

fig = plt.figure()
ax = fig.add_subplot(subplotRows, subplotCols, 1)
ax.plot(stepsPerEpisode)
ax.set_xlabel('Episode')
ax.set_ylabel('Steps')
ax.set_ylim([0, 700])
ax.set_title('Fourier Order ' + str(order) + ' Learning Curve')

plt.show()
