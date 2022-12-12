"""
CSC 449: Programming Assignment 3
SARSA Lambda Fourier Basis Mountain Car

Christian Olson

This program solves the mountain car problem using the true online SARSA(Lambda) with a Fourier Basis.
This is done with bases of order 3, 5, and 7. After computation, the learning curves and -maxQ surface plots are
displayed to show results.

Usage:
$python main.py
"""
from MountainCar import MountainCar
from FourierBasis import FourierBasis
from SarsaLambda import SarsaLambda
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def graphLearningCurve(episodes, order):
    # plot 50 episodes
    plt.plot(episodes[:50])
    plt.ylabel("Number of steps")
    plt.xlabel("Episode")
    title = "SARSA(λ) Learning Curve (Order {} Fourier Basis)".format(order)
    plt.title(title)

    plt.show()

def graphLearningCurves(e3, e5, e7):
    # plot 50 episodes
    plt.plot(e3[:50], label="Order 3")
    plt.plot(e5[:50], label="Order 5")
    plt.plot(e7[:50], label="Order 7")
    plt.legend(loc="upper right")
    plt.ylabel("Number of steps")
    plt.xlabel("Episode")
    plt.title("SARSA(λ) Learning Curves (Order 3, 5, and 7 Fourier Bases)")

    plt.show()

def graphSurface(sim, F, params, order):
    # create state space
    x = np.linspace(-1.2, 0.5, 25)
    y = np.linspace(-0.07, 0.07, 25)
    X, Y = np.meshgrid(x, y)

    # calculate -maxQ(s,a,w)
    z = []
    x = np.ravel(X)
    y = np.ravel(Y)
    for i in range(len(x)):
        normState = MountainCar.normState([x[i], y[i]])
        Qf = F.Q(params, normState, MountainCar.normAction(sim.FORWARD))
        Qi = F.Q(params, normState, MountainCar.normAction(sim.IDLE))
        Qb = F.Q(params, normState, MountainCar.normAction(sim.BACKWARD))
        z.append(-np.max(np.array([Qf, Qi, Qb])))
    z = np.array(z).reshape(X.shape)

    # create 3d figure
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    # plot
    #ax.plot_surface(X, Y, z)
    ax.plot_wireframe(X, Y, z)

    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("-Qmax")
    title = "SARSA(λ) -Qmax (Order {} Fourier Basis at 1000 Episodes)".format(order)
    ax.text2D(0.05, 0.95, title, transform=ax.transAxes)

    plt.show()

def start():
    # create simulation
    sim = MountainCar()

    # create order 3,5, and 7 bases with 3 dimensions
    F3 = FourierBasis(3, 3)
    F5 = FourierBasis(5, 3)
    F7 = FourierBasis(7, 3)

    # learn
    params3, episodes3 = SarsaLambda(sim, F3)
    params5, episodes5 = SarsaLambda(sim, F5)
    params7, episodes7 = SarsaLambda(sim, F7)

    # graph results
    graphLearningCurve(episodes3, 3)
    graphLearningCurve(episodes5, 5)
    graphLearningCurve(episodes7, 7)

    graphLearningCurves(episodes3, episodes5, episodes7)

    graphSurface(sim,F3, params3, 3)
    graphSurface(sim, F5, params5, 5)
    graphSurface(sim, F7, params7, 7)


    return



if __name__ == "__main__":
    start()