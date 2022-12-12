# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 21:25:26 2022

@author: Anoushka Mathews
"""

#!/usr/bin/env python3

from scipy.integrate import solve_ivp
import inverted_pendulum as ip
import json
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import struct
import zmq

## This program is for balancing an inverted pendulum using Q-learning
# temporal differencing off-policy algorithm. 

# Out of 10 tries, this model kinda performs really well on about 3, somewhat 
# well on about 4, and kinda bad on about 3. Emphasis on the words "kinda" and
# "about."

# To run this script, please follow the following steps. These steps assume
# you already have all the dependencies installed. 
# 
# 1) First run the inverted_pendulum_server.py from this folder.
# 2) Then run this script. Make sure the .npy file is in the same directory 
#    as this file. 

# The output should be a balancing pole, and number of attempts/tries. 

# I referenced the following git repo to get started. The code in this file, 
# however, is ALL mine.
#
# https://github.com/HusseinLezzaik/Deep-Q-Learning-for-Inverted-Pendulum
# 

# Sometimes the pendulum goes off the plot window, in that case, I suggest
# waiting patiently for it to go out of bounds, a.k.a die. It will start over
# again.

# Setting the learning rate and exploring to low values, since the model
# has already learned. 
g_alpha = 0.1    # learning_rate
g_epsilon = 0.1   # explore_rate
g_gamma = 0.95    # discount


# Setting more parameter values. I found that this is the most difficult 
# "could-really-slash-should-really-be-optimised-somehow" part of 
# Q-Learning.
max_episodes = 1000
actions = ['right', 'left']
xs = np.linspace(-10, 10, 2)
xdots = np.linspace(-10, 10, 2)
thetas = np.linspace(-0.3, 0.3, 7)
thetadots = np.linspace(-4, 4, 4)

state_dim = (len(xs), len(xdots), len(thetas), len(thetadots))
action_dim = (2)

# If you want to see the model being trained, please use the first 
# initialization of Q. Here I am loading Q with a file that has already has
# correct values for Q. 
# NOTE: It takes about 630,000 episodes to learn with the given parameters.
# Q = np.zeros(state_dim+(action_dim,))
Q = np.load("trained_Q_final.npy")


terminal_state = [0.0, 0.0, 0.0, 0.0]

APPLY_FORCE = 0
SET_STATE = 1
NEW_STATE = 2
ANIMATE = 3
animation_enabled = True

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")


# This function enables animation in the simulator server.
def animate():
    global animation_enabled
    command = ANIMATE
    animation_enabled = 1
    request_bytes = struct.pack('ii', command, animation_enabled)
    socket.send(request_bytes)
    response_bytes = socket.recv()
    return response_bytes


# There is a fixed force of 50 for a "right" action and a -50 for a "left"
# action.
def apply_action(action, u_input=50):
    command = APPLY_FORCE
    if (action == 'right'):
        u = u_input
    elif (action == 'left'):
        u = -u_input
        
    request_bytes = struct.pack('if', command, u)
    socket.send(request_bytes)


# Depending on the value of epsilon, the exploring rate, this function 
# either picks an action randomly or uses an e-greedy policy. 
def get_action(state, epsilon=g_epsilon):
    if (np.random.rand() < epsilon):
        action_index = np.random.randint(0, len(actions))
        action = actions[action_index]
    else:
        x = np.argmax(xs>=state[0])
        xdot = np.argmax(xdots>=state[1])
        theta = np.argmax(thetas>=state[2])
        thetadot = np.argmax(thetadots>=state[3])
        
        action = actions[np.argmax(Q[x][xdot][theta][thetadot])]
        
    return action


# This function initializes a state in the simulator server.
def initialize_state(state):
    command = SET_STATE
    [x, xdot, theta, thetadot] = state
    request_bytes = struct.pack('iffff', command, x, xdot, theta, thetadot)
    socket.send(request_bytes)
    response_bytes = socket.recv()
    return response_bytes
    


def main():
    animate()
    while True:
        for ep in range(max_episodes):
            print("Attempt Number:", ep)
            
            # Initialize the state. I am always starting out at the same
            # value for each episode. 
            # Note: the pole must be started at theta = 0, and thetadot = 0
            # otherwise, the system "may not" work. This is not tested. 
            state = [-2.0, 0.1, 0.0, 0.0]
            initialize_state(state)
            terminate_unexpectedly = False
            
            first= True
            while((state != terminal_state or first) and not(terminate_unexpectedly)):
                
                first = False
                
                # Pick an action
                action= get_action(state)
                apply_action(action)
                
                response_bytes = socket.recv()
                response_command, = struct.unpack('i', response_bytes[0:4])
                
                # We get a new state back based on action we picked in the 
                # last step.
                if response_command == NEW_STATE:
                    
                    x, xdot, theta, thetadot, reward = struct.unpack(
                        'fffff', response_bytes[4:])
                    new_state = [x, xdot, theta, thetadot]
                    
                    
                    # Find the bins/indeces of the original state
                    old_x = np.argmax(xs>=state[0])
                    old_xdot = np.argmax(xdots>=state[1])
                    old_theta = np.argmax(thetas>=state[2])
                    old_thetadot = np.argmax(thetadots>=state[3])
                    
                    
                    # Find the bins/indeces of the new state
                    new_x = np.argmax(xs>=new_state[0])
                    new_xdot = np.argmax(xdots>=new_state[1])
                    new_theta = np.argmax(thetas>=new_state[2])
                    new_thetadot = np.argmax(thetadots>=new_state[3])
                    
                    
                    # Find the action index based on the actions array
                    if(action == 'right'):
                        action_index = 0
                    elif(action == 'left'):
                        action_index= 1
                    
      
                    # If the new state is out of bounds, assign a reward of
                    # -1 and terminate the episode. 
                    # Otherwise, assign new state to original state. 
                    if(abs(new_state[0]) > max(xs) or 
                       abs(new_state[1]) > max(xdots) or 
                       abs(new_state[2]) > max(thetas) or 
                       abs(new_state[3]) > max(thetadots) ):
                        terminate_unexpectedly = True
                        new_reward = -1
                    else:
                        state = new_state.copy()
                        new_reward = reward
                    
                    
                    # original state action value
                    old_state_val = Q[old_x][old_xdot][old_theta][old_thetadot][action_index] 
                    
                    # value of the best action from the new state
                    new_state_max_action = Q[new_x][new_xdot][new_theta][new_thetadot][np.argmax(Q[new_x][new_xdot][new_theta][new_thetadot])]
                    
                    # Da BELLMANN equation..!
                    new_old_state_val = old_state_val + g_alpha*(new_reward+ g_gamma*new_state_max_action-old_state_val) 

                    # Setting new state action value at Q
                    Q[old_x][old_xdot][old_theta][old_thetadot][action_index] = new_old_state_val
                    
                    #print(new_state, reward, new_reward, terminate_unexpectedly)
                    
                else:
                    print("Error: invalid command: ", response_command)
            
            # To save your trained model, make sure to uncomment the following
            # line.
            # np.save("running_training_Q", Q)    


if __name__ == "__main__":
    main()
