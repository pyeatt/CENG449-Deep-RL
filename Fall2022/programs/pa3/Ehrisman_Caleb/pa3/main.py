'''
Author: Caleb Ehrisman
Course- Advanced AI CSC-549
Assignment - Programming Assignment #3 - Mountain Car

Tasks
 - Implement Sarsa(lambda) to solve mountain car problem
 - Use Linear Function Approximation with Fourier Basis functions
 - Show different learning curves for 3rd, 5th, and 7th order Fourier bases
 - Create surface plot of the value function
 - Answer short response question
'''

import gym
import matplotlib

from agent import Agent
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
    parse gathers command line arguments.

    
    :return: a list of all parsed arguments
    """


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', type=str, help='Specify to run simulation or not')
    parser.add_argument('--order', type=int, help='Choose order for fourier basis', default=3)
    parser.add_argument('--num_epochs', type=int, help='Choose number of epochs', default=1000)
    parser.add_argument('--fourier', type=str, help='Choose to use fourier', default=True)
    parser.add_argument('--file', type=str,
                        help='File path to save weights to. Must be given with .npy extension', default='weights.npy')
    parser.add_argument('--train', type=str, help='Choose if training or running', default='True')
    parser.add_argument('--eval', type=str, default='False')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()

    if args.render == "True":
        env = gym.make("MountainCar-v0", render_mode="human")

    else:
        env = gym.make("MountainCar-v0")

    file = args.file

    n = args.num_epochs
    if args.fourier == 'False':
        agent = Agent(env, file, fourier=False, order=3)
    else:
        agent = Agent(env, file, order=3)

    if args.eval == 'False':
        if args.train == 'True':
            rewards, avg, learner = agent.learn(n)
        else:
            rewards, avg = agent.run(n)

        fig, ax = plt.subplots(figsize=(10, 4))
        plt.plot(np.negative(rewards), label='Episode Reward')
        plt.plot(np.negative(avg), label='Running Average')
        ax.set_title("Reward values")
        plt.legend()
        plt.show()

        rewards = []
        base = [3, 5, ]

        rewards.append(avg)

        fig, ax = plt.subplots(figsize=(10, 4))
        plt.plot(np.negative(rewards[0]), label='Base 3')
        plt.plot(np.negative(rewards[1]), label='Base 5')
        plt.plot(np.negative(rewards[2]), label='Base 7')
        ax.set_title("Reward values")
        plt.legend()
        plt.show()

        low = env.observation_space.low
        high = env.observation_space.high
        difference = high - low

        x_axis = np.linspace(low[0], high[0])
        y_axis = np.linspace(low[1], high[1])
        x_axis, y_axis = np.meshgrid(x_axis, y_axis)
        z_axis = np.zeros(x_axis.shape)

        for i in range(0, z_axis.shape[0]):
            for j in range(0, z_axis.shape[1]):
                s = [(x_axis[i, j] - low[0]) / (high[0] - low[0]), (y_axis[i, j] - low[1]) / (high[1] - low[1])]
                (zq, _) = learner.best_action(s)
                z_axis[i, j] = -1.0 * zq

            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot_surface(x_axis, y_axis, z_axis, cmap=matplotlib.cm.get_cmap("magma"))
            ax.set_xlabel('position')
            ax.set_ylabel('velocity')
            ax.set_title('Cost Function for Order - ' + str(n))
            plt.show()

    else:
        rewards = []
        base = [3, 5, 7]
        learner = []
        for i in range(3):
            agent = Agent(env, file, order=base[i])
            reward, avg, temp_learner = agent.learn(1000)
            rewards.append(avg)
            learner.append(temp_learner)

        fig, ax = plt.subplots(figsize=(10, 4))
        plt.plot(np.negative(rewards[0]), label='Base 3')
        plt.plot(np.negative(rewards[1]), label='Base 5')
        plt.plot(np.negative(rewards[2]), label='Base 7')
        ax.set_title("Reward values")
        plt.legend()
        plt.show()

        low = env.observation_space.low
        high = env.observation_space.high
        difference = high - low

        x_axis = np.linspace(low[0], high[0])
        y_axis = np.linspace(low[1], high[1])
        x_axis, y_axis = np.meshgrid(x_axis, y_axis)
        z_axis = np.zeros(x_axis.shape)

        for b in range(3):
            for i in range(0, z_axis.shape[0]):
                for j in range(0, z_axis.shape[1]):
                    s = [(x_axis[i, j] - low[0]) / (high[0] - low[0]), (y_axis[i, j] - low[1]) / (high[1] - low[1])]
                    (zq, _) = learner[b].best_action(s)
                    z_axis[i, j] = -1.0 * zq

            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot_surface(x_axis, y_axis, z_axis, cmap=matplotlib.cm.get_cmap("magma"))
            ax.set_xlabel('position')
            ax.set_ylabel('velocity')
            ax.set_title('Cost Function for Order - ' + str(base[b]))
            plt.show()





