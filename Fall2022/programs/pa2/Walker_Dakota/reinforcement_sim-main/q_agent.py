#!/usr/bin/env python3

from scipy.integrate import solve_ivp
import inverted_pendulum as ip
import json
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import random
import struct
import zmq

gamma = 0.99
learning_rate = 0.2
epsilon = 0.15
state_space = 4
action_space = 7
bin_size = 15
max_steps = 1000

#create the q_table that will store the q values and bins to store possible states
def Qtable(q_file, state_space,action_space):
    
    bins = [np.linspace(-5,5,bin_size),
            np.linspace(-5,5,bin_size),
            np.linspace(-np.pi,np.pi,bin_size),
            np.linspace(-10,10,bin_size)]
    
    q_table = np.load(q_file)
    #q_table = np.zeros([bin_size] * state_space + [action_space])
    
    return q_table, bins

#find the bins that the current state values belong in to represent the current state
def binning(state, bins):
    index = []
    for i in range(len(state)): index.append(np.digitize(state[i],bins[i]) - 1)
    
    return tuple(index)
''

#e_greedy to choose next step
    #epsilon vs best action
def e_greedy(q_table, state, epsilon):
    if np.random.uniform(0,1) < epsilon:
        action = random.randrange(0,7)  #randomly select 0 to 6
    else:
        action = np.argmax(q_table[state])
        
    return action

#update the q value in the q_table
def update_q(q_table, c_state, n_state, action, reward):
    #next best action's value
    future_q = np.max(q_table[n_state])
    #current state's q value
    current_q = q_table[c_state+(action,)]
    #calc new q value
    new_q = (reward + gamma * future_q) - current_q
    
    q_table[c_state+(action,)] += learning_rate * new_q
    return q_table
 

def main():
    APPLY_FORCE = 0
    SET_STATE = 1
    NEW_STATE = 2
    ANIMATE = 3
    action = 0
    reward = 0
    trials = -1
    score = 0
    
    #location of Q values from training
    q_file = 'q_table.npy'

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    
    #open file to load Q table
    fout = open(q_file, 'r+')
    #get Q table and state space bins
    q_table, bins = Qtable(q_file, state_space, action_space)
    fout.close()
    
    #reopen file to save Q table
    fout = open(q_file, 'w+')
    
    animation_enabled = True
    count = 0
    while True:
        if  count % max_steps == 0:
            # toggle animation
            command = ANIMATE
            animation_enabled = not animation_enabled
            request_bytes = struct.pack('ii', command, animation_enabled)
            socket.send(request_bytes)

        elif count % max_steps == 1:
            #output the trial's number and total reward score
            trial = int(count/max_steps)
            if trial % 100 == 0 or score > -1.0:
                np.save(q_file, q_table)
                print("Trial:", trial, " Score:", score)
                
            # reset the state
            score = 0
            command = SET_STATE
            x = -2.0
            xdot = 0.0
            theta = 0.2
            thetadot = 0.0
            new_state = [x, xdot, theta, thetadot]
            
            #bin starting state
            c_bin_state = binning(new_state, bins)
            
            request_bytes = struct.pack(
                'iffff', command, x, xdot, theta, thetadot)
            socket.send(request_bytes)

        else:
            #determine best action using e-greedy
            action = e_greedy(q_table, c_bin_state, epsilon)
            #apply the action selected
            command = APPLY_FORCE
            request_bytes = struct.pack('if', command, ((action-3)*4))
            socket.send(request_bytes)

        count += 1

        response_bytes = socket.recv()
        response_command, = struct.unpack('i', response_bytes[0:4])

        if response_command == NEW_STATE:
            x, xdot, theta, thetadot, reward = struct.unpack(
                'fffff', response_bytes[4:])
            new_state = [x, xdot, theta, thetadot]
            
            #terminate episode if out of x bounds
            if new_state[0] < -5 or new_state[0] > 5:
               count = (int(count/max_steps) * max_steps) + max_steps + 1
            
            #update trial's score
            score += reward
            
            #bin current state
            n_bin_state = binning(new_state, bins)
            
            #update q_table
            q_table = update_q(q_table, c_bin_state, n_bin_state, action, reward)
            
            #update current state with new state
            c_bin_state = n_bin_state
            
            #print(new_state, reward)
        elif response_command == ANIMATE:
            animation_enabled, = struct.unpack('i', response_bytes[4:])
        else:
            print("Error: invalid command: ", response_command)
            
    fout.close()

if __name__ == "__main__":
    main()
    

