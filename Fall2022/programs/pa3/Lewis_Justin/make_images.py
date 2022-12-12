import os
from pathlib import Path
import glob
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib

load_directory = "results"
save_directory = "writeup/images/"

x = np.linspace(-1.5, .5, 128)
y = np.linspace(-.7, .7, 128)
X, Y = np.meshgrid(x, y)

x_1k = np.array(range(1000))

def make_image_3d(z, order, fileName):
    fig = plt.figure(figsize=(13,10))
    ax = plt.axes(projection="3d")
    ax.set_title("Order = " + order)
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.plot_surface(X, Y, z)
    plt.savefig(save_directory + fileName)

def render_value(iteration_counts, ax, color):
    ax.plot(x_1k, iteration_counts, color)

def render_values(iteration_counts, ax, labels):
    ax.set_title("All together")
    colors = ['r-', 'g-', 'b-']
    lines = []
    for i, order_values in enumerate(iteration_counts):
        res, = ax.plot(x_1k, order_values, colors[i], label=labels[i])
        lines.append(res)
    ax.legend(handles=lines)

def make_all_values(all_values, labels, fileName):
    fig, axs = plt.subplots(2, 2, figsize=(13,10))
    axs = axs.flatten()
    plt.xlim((-10, 1000))
    for ax in axs:
        ax.set_yscale("log")
        ax.set_ylim((100, 1000))
        ax.set_yticks((100, 200, 300, 400, 600, 1000)) # needed so they show up in non-scientific-notation
        ax.set_xlabel("Episode Number")
        ax.set_ylabel("Steps per Episode")
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    render_values(values, axs[0], labels)
    colors = ['r-', 'g-', 'b-']
    for i, ax in enumerate(axs[1:]):
        ax.set_title(labels[i])
        render_value(values[i], ax, colors[i])

    plt.savefig(save_directory + fileName)

if __name__ == "__main__":
    all_values = glob.glob(load_directory + "/*values")
    all_iterations = glob.glob(load_directory + "/*iterations")
    for value in all_values:# this will make the surface plot of value function
        order = value[-8:-7]
        z = np.loadtxt(value)  
        make_image_3d(z, order, "surfaceOrder" + order)
    values = [np.loadtxt(value) for value in all_iterations]
    labels = [title[-13:-11] for title in all_iterations] # really only works for single digit order
                                                          # I should bring regex into it
    labels = ["Order = " + label[1] for label in labels]
    make_all_values(values, labels, "2by2")

