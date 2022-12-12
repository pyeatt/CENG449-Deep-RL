"""
This file is to just use the SARSA-Lambda algorithm in whatever way I please

I've been using it to play around with parameters to see how they affect the 
algorithm
"""
from sarsa_lambda import SarsaLambda
from fourier_basis import FourierBasis
from polynomial_basis import PolynomialBasis
import numpy as np

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib

EPISODES = 1000
x_1k = np.array(range(EPISODES))
def draw(x, y, z):
    fig = plt.figure(figsize=(15,15))
    ax = plt.axes(projection="3d")
    ax.plot_surface(x, y, z)
    plt.show()

def draw_learning_curve(episode_steps):
    plt.plot(x_1k, episode_steps)



if __name__ == "__main__":
    fig = plt.figure(figsize=(8,8))
    gammas = [1, .99]
    order=3
    dimensions = 3

    basis = FourierBasis(dimensions, order)
    all_iteration_counts = []
    for gamma in gammas:
        sl = SarsaLambda(basis.apply, c=basis.getC(), d=dimensions, order=order, gamma=gamma)
        iteration_counts = []
        for k in range(EPISODES):
            res = sl.run_episode()
            iteration_counts.append(res)
            if k % 200 == 0:
                print("\tEpisode: ", k, res)
        all_iteration_counts.append(iteration_counts)

    print(np.shape(all_iteration_counts))
    for i, gamma in enumerate(gammas):
        plt.clf()
        plt.yscale("log")
        plt.yticks((100, 200, 300, 400, 600, 1000))
        plt.xlabel("Episode Number")
        plt.ylabel("Steps per Episode")
        plt.ylim((100, 1000))
        plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        draw_learning_curve(all_iteration_counts[i])
        plt.savefig("writeup/images/gamma" + str(i))

