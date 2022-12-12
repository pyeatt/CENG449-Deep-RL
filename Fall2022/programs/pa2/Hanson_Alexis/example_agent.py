#!/usr/bin/env python3

import numpy as np
import struct
import zmq
import random

N = 7   #program bugs out with N = even, could not be bothered to fix
N_HALF = np.floor(N/2).astype(int) #keeping floor(N/2) around is handy instead of doing recalculations
ACTIONS = [-10,0,10]
ALPHA = 0.3
GAMMA = 0.87


APPLY_FORCE = 0
SET_STATE = 1
NEW_STATE = 2
ANIMATE = 3

MAX_PATIENCE = 5000 #added 2 zeros from its value during the hyperparameter search that decided that my alpha value was waaay too high

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

animation_enabled = True
count = 0

def bin_states(exact_states):
    signs = (exact_states / np.abs(exact_states)).astype(int)
    #pair of arrays to determine exact binning of values, the x translations are linear across the whole state space, but the y ones are highly focused around the center and cap out at the states which are generally unrecoverable for it anyways
    linear_scaling = 0.5*np.array([5,1.5,1.2,2.5])
    exp_scaling = np.array([1,1,0.5,0.5])
    abs_binned = np.minimum(np.floor(np.power(np.abs(exact_states),exp_scaling)*N/linear_scaling),N_HALF).astype(int) #process and cap to fit in lookup
    return [N_HALF,N_HALF,N_HALF,N_HALF] + signs * abs_binned


try:
    with open('values.npy', 'rb') as f:
        Q = np.load(f)
except:
    Q = np.zeros([N,N,N,N,len(ACTIONS)])
max_lifetime = 0
patience = 0
episodes = 0
while patience < MAX_PATIENCE:
    if count == 0:

        # toggle animation
        command = ANIMATE
        animation_enabled = not animation_enabled
        request_bytes = struct.pack('ii', command, animation_enabled)
        socket.send(request_bytes)
        lifetime = count
    elif count == 1:
        print("resetting:")
        # reset the state
        command = SET_STATE
        starting_state = [2*(random.random()-0.5),(random.random()-0.5),0.4*(random.random()-0.5),0.7*(random.random()-0.5)]
        action = 1
        old_x,old_xdot,old_theta,old_thetadot = bin_states(starting_state)
        #x,xdot,theta,thetadot
        request_bytes = struct.pack(
            'iffff', command, *starting_state)
        socket.send(request_bytes)

    else:
        command = APPLY_FORCE
        u = ACTIONS[action]
        request_bytes = struct.pack('if', command, u)
        socket.send(request_bytes)

    count += 1

    response_bytes = socket.recv()
    response_command, = struct.unpack('i', response_bytes[0:4])

    if response_command == NEW_STATE:
        x, xdot, theta, thetadot, reward = struct.unpack(
            'fffff', response_bytes[4:])
        new_state = np.array([x, xdot, theta, thetadot])
        
        #check whether things are out of bounds, I gave them a buffer of negative rewards before the reset because it seemed to converge faster
        if np.abs(x) > 2.5 or np.abs(theta) > 1:
            reward =-1
        if np.abs(x) > 3 or np.abs(theta) > 1.5:
            if max_lifetime < count:
                max_lifetime = count
                patience = 0                        #I initially added patience for a hyperparameter search but I assume you don't want to run that
            else:
                patience += 1 
            episodes +=1
            count = 0
        
        binned_states = bin_states(new_state)
        print(new_state, reward)
        print(f'Binned as: {binned_states}')
        print(f'Value:{Q[old_x,old_xdot,old_theta,old_thetadot,:]}')
        print(f'Action: {action},Count: {count}')
        #update states
        x, xdot, theta, thetadot = binned_states
        #I have literally seen the bellman equation show up in my dreams. This is your fault.
        Q[old_x,old_xdot,old_theta,old_thetadot,action] += ALPHA * (reward + GAMMA * np.max(Q[x,xdot,theta,thetadot, :])- Q[old_x,old_xdot,old_theta,old_thetadot,action])
        #is there a concise way to make Q[*old_state,a] work as it would if I was calling f(*old_state,a)
        #trained with half baked adaptive e-greedy, switched to greedy for submission
        #if random.random() < (1-(0.3/episodes)  
        action = np.argmax(Q[x,xdot,theta,thetadot])
        #else:
        #    action = random.randint(0,len(ACTIONS)-1)
            
        old_x,old_xdot,old_theta,old_thetadot = binned_states
        
    elif response_command == ANIMATE:
        animation_enabled, = struct.unpack('i', response_bytes[4:])
    else:
        print("Error: invalid command: ", response_command)
with open('values.npy', 'rb') as f:
    np.save(f,Q)    #save weights once done training
