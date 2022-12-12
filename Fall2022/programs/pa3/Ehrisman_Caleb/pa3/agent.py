'''
Author: Caleb Ehrisman
Course- Advanced AI CSC-549
Assignment - Programming Assignment #3 - Mountain Car

This file contains the code to execute training of either sarsa implementation based on params

All functions needed by solely the agent are included as member functions of class Agent
'''
import numpy as np
from fourier_basis import FourierBasis
from sarsalambdaFA import SarsaLambdaFA
from sarsa import Sarsa
import os.path

ALPHA = 0.0001
GAMMA = 1
EPSILON = 0.5
LAMBDA = 0.9


class Agent:

    def __init__(self, environment, file, fourier=True, order=3, runs=1, gamma=0.001):
        """
                init is the constructor for the Agent class.

                :param environment
                :return None
        """
        self.runs = runs
        self.order = order
        self.env = environment
        self.gamma = gamma
        self.num_actions = self.env.action_space.n
        self.state_dims = self.env.observation_space.shape[0]
        self.fourier = fourier
        self.epoch_rewards = []
        self.epoch_rewards_table = {'ep': [], 'avg': [], 'min': [], 'max': []}
        self.epoch_max_pos = []
        self.file = file

    def learn(self, num_epochs):
        """
            Agent.learn does the actual stepping through and exploring the environment and then updates the Q_table if
            using traditional SARSA and updates the weight and lambda vectors is using a fourier basis

            :param num_epochs
            :return None
            """
        for run in range(0, self.runs):
            fb = FourierBasis(state_space=self.env.observation_space.shape[0], order=self.order)
            if self.fourier:
                learner = SarsaLambdaFA(fa=fb, num_actions=self.num_actions, alpha=0.0001, epsilon=0.8)
            else:
                learner = Sarsa(environment=self.env)

            for i in range(num_epochs):

                learner.epsilon *= .99
                if self.fourier:
                    learner.z = np.zeros(learner.w.shape)
                state, _ = self.env.reset()
                if self.fourier:
                    state = (state - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low)
                else:
                    state = learner.discretized_env_state(state)
                    learner.E_table = learner.create_q_table()
                # steps = 0
                action = learner.action(state)
                done = False
                reward_sum = 0

                while not done:
                    next_state, reward, done, info, _ = self.env.step(action)
                   # reward += 1
                   # if done:
                   #     reward = 100
                    if self.fourier:
                        next_state = (next_state - self.env.observation_space.low) / (
                                    self.env.observation_space.high - self.env.observation_space.low)
                    else:
                        next_state = learner.discretized_env_state(next_state)

                    next_action = learner.action(next_state)

                    learner.update(state, action, reward, next_state, done, next_action)

                    # steps += 1
                    state = next_state
                    action = next_action
                    reward_sum += reward

                #  Append max position data and reward data for evaluation
                self.epoch_rewards.append(reward_sum)

                self.terminal_output(i)
        np.save(self.file, learner.w)
        return self.epoch_rewards, self.epoch_rewards_table['avg'], learner

    def run(self, num_epochs):
        """
            Agent.run uses a pre-trained set of weights to greedily choose actions.

            :param num_epochs
            :return None
            """

        if os.path.exists(self.file):
            w = np.load(self.file)
        else:
            print("Error loading file. Not found.")
            return

        fb = FourierBasis(state_space=self.env.observation_space.shape[0], order=self.order)
        learner = SarsaLambdaFA(fa=fb, num_actions=self.num_actions, alpha=0.0, epsilon=0.0)
        learner.w = w

        for i in range(num_epochs):
            state, _ = self.env.reset()

            state = (state - self.env.observation_space.low) / (
                        self.env.observation_space.high - self.env.observation_space.low)

            action = learner.action(state)
            done = False
            reward_sum = 0

            while not done:
                next_state, reward, done, info, _ = self.env.step(action)
                print(reward)
                next_state = (next_state - self.env.observation_space.low) / (
                        self.env.observation_space.high - self.env.observation_space.low)

                next_action = learner.action(next_state)
                action = next_action
                reward_sum += reward

            #  Append max position data and reward data for evaluation
            self.epoch_rewards.append(reward_sum)

            self.terminal_output(i)

        return self.epoch_rewards, self.epoch_rewards_table['avg']

    def terminal_output(self, i):
        # Terminal Output for stats of each epoch
        avg_reward = sum(self.epoch_rewards[-10:]) / len(self.epoch_rewards[-10:])
        self.epoch_rewards_table['ep'].append(i)
        self.epoch_rewards_table['avg'].append(avg_reward)
        self.epoch_rewards_table['min'].append(min(self.epoch_rewards[:]))
        self.epoch_rewards_table['max'].append(max(self.epoch_rewards[:]))

        print(f"Epoch - {i}\t| avg: {avg_reward:.2f}\t| min: {min(self.epoch_rewards[-1:]):.2f}"
              f"\t| max: {max(self.epoch_rewards[-1:]):.2f}")
