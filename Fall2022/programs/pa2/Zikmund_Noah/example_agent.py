# Noah Zikmund
# CENG 449
# Dr. Pyeatt
# 10 October 2022

# PA2: Inverted Pendulum Problem

# The problem is to implement a pendulum controller using Tabular TD. Sarsa or Q-learning will work
# The solution to this program was simulated using Johnathan Matthews's simulator found here at:
# https://github.com/ben-rambam/reinforcement_sim

# Please note that this code file mostly consists of Johnathan's starter code

# For Sarsa, a state-action value pairs function is desired
# Algorithm parameters: step size ↵ 2 (0, 1], small " > 0
    # Initialize Q(s, a), for all s within S+, a within A(s), arbitrarily except that Q(terminal, .)=0

    # Loop for each episode:
    # Initialize S
    # Choose A from S using policy derived from Q (e.g., "-greedy)
    # Loop for each step of episode:
    # Take action A, observe R, S'
    # Choose A' from S' using policy derived from Q (e.g., "-greedy)
    # Q(S, A) <- Q(S, A) + alpha[R + γQ(S', A') - Q(S, A)]
    # S <- S'
    # A <- A'
    # until S is terminal

from scipy.integrate import solve_ivp
import inverted_pendulum as ip
import json
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import struct
import zmq

X_bin = "10"          # X-direction
Xd_bin = "10"         # Velocity of X-direction
Th_bin = "20"         # Theta or angle of pendulum
Thd_bin = "20"        # Velocity at with theta changes
take_action = '1'     # Left = 0, Right = 1
alpha = 0.02
gamma = 1

# Declare 5D array to hold state-action value function
Q = np.zeros(106722).reshape((11,11,21,21,2))

# The simulator defines the bounds for the x-direction to be from -5 to 5, so this function
# maps those values to bins ranging from 0 - 10 for an appropriate array index
def get_x_bin(current_x):
    if(current_x < -4):
        retBin = '0'
    elif(current_x >= -4 and current_x < -3):
        retBin = '1'
    elif(current_x >= -3 and current_x < -2):
        retBin = '2'
    elif(current_x >= -2 and current_x < -1):
        retBin = '3'
    elif(current_x >= -1 and current_x < 0):
        retBin = '4'
    elif(current_x >= 0 and current_x < 1):
        retBin = '5'
    elif(current_x >= 1 and current_x < 2):
        retBin = '6'
    elif(current_x >= 2 and current_x < 3):
        retBin = '7'
    elif(current_x >= 3 and current_x < 4):
        retBin = '8'
    elif(current_x >= 4 and current_x < 5):
        retBin = '9'
    elif(current_x > 5):
        retBin = "10"
    return retBin

# The simulator defines the bounds for the velocity to be appx. -4 to 4. However,
# the velocities are hardly every greater than 2 or -2, so this function maps those
# values to bins in a non-linear fashion, providing more resolution to slower velocities
def get_xd_bin(current_x_dot):
    if(current_x_dot > -4 and current_x_dot <= -3):
        retBin = '0'
    elif(current_x_dot > -3 and current_x_dot <= -2):
        retBin = '1'
    elif(current_x_dot > -2 and current_x_dot <= -1):
        retBin = '2'
    elif(current_x_dot > -1 and current_x_dot <= -0.5):
        retBin = '3'
    elif(current_x_dot > -0.5 and current_x_dot <= 0):
        retBin = '4'
    elif(current_x_dot > 0 and current_x_dot <= 0.5):
        retBin = '5'
    elif(current_x_dot > 0.5 and current_x_dot <= 1):
        retBin = '6'
    elif(current_x_dot > 1 and current_x_dot <= 1.5):
        retBin = '7'
    elif(current_x_dot > 1.5 and current_x_dot <= 2.5):
        retBin = '8'
    elif(current_x_dot > 2.5 and current_x_dot <= 3.5):
        retBin = '9'
    elif(current_x_dot > 3.5):
        retBin = "10"
    return retBin
    
# The simulator defines the bounds for the theta value to be from 0 - 6.4 where 
# 0 and 6.4 are both close representations of the pole standing 90 degrees straight up.
# This function maps those values into bins for appropriate array indexing
def get_theta_bin(current_theta):
    if(current_theta > 0 and current_theta < 0.32):
        retBin = '0'
    elif(current_theta > 0.32 and current_theta <= 0.64):
        retBin = '1'
    elif(current_theta > 0.64 and current_theta <= 0.96):
        retBin = '2'
    elif(current_theta > 0.96 and current_theta <= 1.28):
        retBin = '3'
    elif(current_theta > 1.28 and current_theta <= 1.6):
        retBin = '4'
    elif(current_theta > 1.6 and current_theta <= 1.92):
        retBin = '5'
    elif(current_theta > 1.92 and current_theta <= 2.24):
        retBin = '6'
    elif(current_theta > 2.24 and current_theta <= 2.56):
        retBin = '7'
    elif(current_theta > 2.56 and current_theta <= 2.88):
        retBin = '8'
    elif(current_theta > 2.88 and current_theta <= 3.2):
        retBin = '9'
    elif(current_theta > 3.2 and current_theta <= 3.52):
        retBin = "10"
    elif(current_theta > 3.52 and current_theta <= 3.84):
        retBin = "11"
    elif(current_theta > 3.84 and current_theta <= 4.16):
        retBin = "12"
    elif(current_theta > 4.16 and current_theta <= 4.48):
        retBin = "13"
    elif(current_theta > 4.48 and current_theta <= 4.8):
        retBin = "14"
    elif(current_theta > 4.8 and current_theta <= 5.12):
        retBin = "15"
    elif(current_theta > 5.12 and current_theta <= 5.44):
        retBin = "16"
    elif(current_theta > 5.44 and current_theta <= 5.76):
        retBin = "17"
    elif(current_theta > 5.76 and current_theta <= 6.08):
        retBin = "18"
    elif((current_theta > 6.08 and current_theta <= 6.4) or current_theta > 6.4 or current_theta < 0):
        retBin = "19"
    return retBin

# The simulator defines the bounds for the theta dot value to be from -10 to 10 appx.
# This function maps those values into bins for appropriate array indexing
def get_thetad_bin(current_theta_dot):
    retBin = '20'
    if(current_theta_dot < -9 and current_theta_dot >= -10):
        retBin = '0'
    elif(current_theta_dot < -8 and current_theta_dot >= -9):
        retBin = '1'
    elif(current_theta_dot < -7 and current_theta_dot >= -8):
        retBin = '2'
    elif(current_theta_dot < -6 and current_theta_dot >= -7):
        retBin = '3'
    elif(current_theta_dot < -5 and current_theta_dot >= -6):
        retBin = '4'
    elif(current_theta_dot < -4 and current_theta_dot >= -5):
        retBin = '5'
    elif(current_theta_dot < -3 and current_theta_dot >= -4):
        retBin = '6'
    elif(current_theta_dot < -2 and current_theta_dot >= -3):
        retBin = '7'
    elif(current_theta_dot < -1 and current_theta_dot >= -2):
        retBin = '8'
    elif(current_theta_dot < 0 and current_theta_dot >=-1):
        retBin = '9'
    elif(current_theta_dot < 1 and current_theta_dot >= 0):
        retBin = "10"
    elif(current_theta_dot < 2 and current_theta_dot >= 1):
        retBin = "11"
    elif(current_theta_dot < 3 and current_theta_dot >= 2):
        retBin = "12"
    elif(current_theta_dot < 4 and current_theta_dot >= 3):
        retBin = "13"
    elif(current_theta_dot < 5 and current_theta_dot >= 4):
        retBin = "14"
    elif(current_theta_dot < 6 and current_theta_dot >= 5):
        retBin = "15"
    elif(current_theta_dot < 7 and current_theta_dot >= 6):
        retBin = "16"
    elif(current_theta_dot < 8 and current_theta_dot >= 7):
        retBin = "17"
    elif(current_theta_dot < 9 and current_theta_dot >= 8):
        retBin = "18"
    elif(current_theta_dot <= 10 and current_theta_dot >= 9):
        retBin = "19"
    elif(current_theta_dot > 10 and current_theta_dot < -10):
        retBin = "20"

    return retBin

def calculate_Q(reward):
    global Q, X_bin, Xd_bin, Th_bin, Thd_bin, take_action, alpha, gamma

    nextQ = int(calculate_next_Q())
    a = int(X_bin)
    b = int(Xd_bin)
    c = int(Th_bin)
    d = int(Thd_bin)
    e = int(take_action)
    Q[a, b, c, d, e] = Q[a, b, c, d, e] + (alpha * (reward + (gamma*nextQ) - Q[a, b, c, d, e]))

def get_action():
    global Q, X_bin, Xd_bin, Th_bin, Thd_bin, take_action

    a = int(X_bin)
    b = int(Xd_bin)
    c = int(Th_bin)
    d = int(Thd_bin)
    
    maximum =  Q[a, b, c, d, 1]
    take_action = '1'

    if(Q[a, b, c, d, 0] > maximum):
        maximum =  Q[a, b, c, d ,0]
        take_action = '0'
    
    return take_action

def calculate_next_Q():
    global Q, X_bin, Xd_bin, Th_bin, Thd_bin, take_action

    a = int(X_bin)
    b = int(Xd_bin)
    c = int(Th_bin)
    d = int(Thd_bin)

    maximum = Q[a, b, c, d, 1]
    take_action = '1'

    if(Q[a, b, c, d, 0] > maximum):
        maximum =  Q[a, b, c, d, 0]
        take_action = '0'

    return maximum

def main():
    global Q, X_bin, Xd_bin, Th_bin, Thd_bin

    APPLY_FORCE = 0
    SET_STATE = 1
    NEW_STATE = 2
    ANIMATE = 3

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    animation_enabled = True
    count = 0
    while True:
        if count % 1000 == 0:

            # toggle animation
            command = ANIMATE
            animation_enabled = not animation_enabled
            request_bytes = struct.pack('ii', command, animation_enabled)
            socket.send(request_bytes)

        elif count % 1000 == 1:
            # reset the state
            command = SET_STATE
            x = -2.0
            xdot = 0.0
            theta = 0.2
            thetadot = 0.0
            request_bytes = struct.pack(
                'iffff', command, x, xdot, theta, thetadot)
            socket.send(request_bytes)

        # Replace with my actions
        else:
            command = APPLY_FORCE
            temp = get_action()

            if(temp == '0'):
                u = -0.3
            elif(temp == '1'):
                u = 0.3

            request_bytes = struct.pack('if', command, u)
            socket.send(request_bytes)

        count += 1

        response_bytes = socket.recv()
        response_command, = struct.unpack('i', response_bytes[0:4])

        # Update with my Qs
        if response_command == NEW_STATE:
            x, xdot, theta, thetadot, reward = struct.unpack(
                'fffff', response_bytes[4:])

            # ******************** MY CODE ************************
            X_bin = get_x_bin(x)
            Xd_bin = get_xd_bin(xdot)
            Th_bin = get_theta_bin(theta)
            Thd_bin = get_thetad_bin(thetadot)

            calculate_Q(reward)
            # ******************* END MY CODE *********************

            new_state = [x, xdot, theta, thetadot]
            print(new_state, reward, take_action, Q[int(X_bin), int(Xd_bin), int(Th_bin), int(Thd_bin), int(take_action)])
        elif response_command == ANIMATE:
            animation_enabled, = struct.unpack('i', response_bytes[4:])
        else:
            print("Error: invalid command: ", response_command)

if __name__ == "__main__":
    main()

