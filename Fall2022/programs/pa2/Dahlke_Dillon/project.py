'''**********************************************************
* Dillon Dahlke
*
* CSC 449 - Programming Assignment 2
*
*Python Dependencies Needed:
*   gym
*   numpy
*   matplotlib
*
*This program implements Q learning to solve the CartPole
* problem. It uses the environment provided in Gym, a 
* a library for reinforcement learning environments.
*
*************************************************************'''

import gym
import numpy as np


#Initialize the environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

#A rate of learning from 0 to 1
LEARNING_RATE = 0.1
#Discount factor gamma
GAMMA = 0.95
#Number of episodes to run
EPISODES = 1000

#Discretize our observation space
OS_SIZE_DISCRETE = [80] * len(env.observation_space.high)
#Get rid of infinite velocities
lesser_os_hi = env.observation_space.high
lesser_os_hi[1] = 100
lesser_os_hi[3] = 100

lesser_os_lo = env.observation_space.low
lesser_os_lo[1] = -100
lesser_os_lo[3] = -100
#Let's determine how big our "buckets" really are
discrete_os_win_size = (lesser_os_hi - lesser_os_lo) / OS_SIZE_DISCRETE
print(discrete_os_win_size)

#Create a table for every single observation combination with a corresponding value for each action
#(This can be thought of as a three dimensional table)
#Initialize the table with random values to signify random actions in the early stages of development
qtable = np.random.uniform(low=-2, high=0, size = (OS_SIZE_DISCRETE + [env.action_space.n]))

print(qtable.shape)
#Helper function to convert a continuous state to a discrete state
def to_discrete(state):
    d_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(d_state.astype(np.int))

for episode in range(EPISODES):
    print(episode)
    #Reset the environment
    state = env.reset()
    state = state[0]
    #Get a value for the initial state
    #action = 0
    #new_state, reward, done, too_long, _ = env.step(action)
    discrete_state = to_discrete(state)

    done = False
    too_long = False
    #Loop through while the episode is not done
    while not done or too_long:
        #Get a new action based on the greedy property of bellman
        action = np.argmax(qtable[discrete_state])
        new_state, reward, done, too_long, _ = env.step(action)

        #Get a discrete value for the new state
        new_discrete_state = to_discrete(new_state)
        #render the current state to visualize
        env.render()
        if not done:
            max_future_q = np.max(qtable[new_discrete_state])
            #get the q value of the current state
            #Each state has 3 values so an offset is used
            current_q = qtable[discrete_state + (action, )]
            #implementation of the bellman equation for q learning
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + GAMMA * max_future_q)
            #Update the q table with the new q value
            qtable[discrete_state+(action, )] = new_q

        discrete_state = new_discrete_state
    env.close()

env.close()
env = gym.make("CartPole-v1", render_mode="human")
env.reset()
#Loop through while the episode is not done
while not done or too_long:
    #Get a new action based on the greedy property of bellman
    action = np.argmax(qtable[discrete_state])
    new_state, reward, done, _, _ = env.step(action)
    #Get a discrete value for the new state
    new_discrete_state = to_discrete(new_state)
    discrete_state = new_discrete_state