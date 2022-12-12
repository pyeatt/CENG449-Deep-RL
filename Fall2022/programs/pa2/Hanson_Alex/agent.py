'''
Alex Hanson

Python version: Python 3.9.0


## Running pre trained
> python agent.py -a -r -l trained_q
'''

#!/usr/bin/env python3

from scipy.integrate import solve_ivp
import inverted_pendulum as ip
import json
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import struct
import zmq


import random # todo maybe remove
import numpy as np
import math

# Handle command line args
import argparse
msg = "description"
parser = argparse.ArgumentParser(description = msg)
parser.add_argument("-a", "--animate", action='store_true', dest='animate', help="Enables animation on server.")
parser.add_argument("-e", "--episodes", default=-1, dest="episodes", help="Number of episodes to run.", type=int)
parser.add_argument("-s", "--save", type=str, dest="qfileout", const="q_values", help="File to save q values too. Defaults to 'q_values'", nargs="?")
parser.add_argument("-l", "--load", type=argparse.FileType('rb'), dest="qfilein", help="The file to load q values from")
parser.add_argument("-r", "--run", action='store_true', dest='run', help="Keeps Constant epsilon and alpha of 0.1. When omited epsilon and alpha start at 0.99 and reduce to 0.1.")
args = parser.parse_args()

SAVE = False
LOAD = False

if args.qfileout is not None:
    SAVE = True
    outfile = args.qfileout
if args.qfilein is not None:
    LOAD = True
    infile = args.qfilein



LEFT_BOUND, RIGHT_BOUND = -5, 5
INIT_X, INIT_XDOT, INIT_THETA, INIT_THETADOT = 0.0, 0.0, 0.05, 0.0 # 0.2
APPLY_FORCE, SET_STATE, NEW_STATE, ANIMATE = 0,1,2,3

LEFT, ZERO, RIGHT, HEAVY_LEFT, HEAVY_RIGHT = -0.5, 0.0, 0.5, -3.0, 3.0

# discount factor
gamma = 1.0 # 0.99 #1.0
# exploration rate
epsilon = 0.99#0.9
# learning rate
alpha = 0.99 # 0.2


# range of values
x_min, x_max = -1.0, 1.0
xdot_min, xdot_max = -0.8, 0.8
theta_min, theta_max = -0.6, 0.6
thetadot_min, thetadot_max = -2.0, 2.0

# buckets
x_buckets = 3
xdot_buckets = 3
theta_buckets = 11
thetadot_buckets = 6

num_buckets = x_buckets * xdot_buckets * theta_buckets * thetadot_buckets
#162 # 3 * 3 * 3 * 6




if args.run:
    epsilon = 0.1
    alpha = 0.2


def main():

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    # enable or disable animation on server
    animation_enabled = args.animate
    command = ANIMATE
    request_bytes = struct.pack('ii', command, animation_enabled)
    socket.send(request_bytes)
    response_bytes = socket.recv()
    response_command, = struct.unpack('i', response_bytes[0:4])


    count = 0
    episode_count = 0
    num_episodes = args.episodes

    x, xdot, theta, thetadot, reward = 0, 0, 0, 0, 0
    actions = [HEAVY_LEFT, ZERO, HEAVY_RIGHT]#[LEFT, ZERO, RIGHT, HEAVY_LEFT, HEAVY_RIGHT]#[HEAVY_LEFT, ZERO, HEAVY_RIGHT]
    Q = np.zeros([num_buckets, len(actions)])
    if LOAD:
        Q = np.load(infile)
        print("Q loaded")
        print(Q)
    # print(Q)


    def initState():
        nonlocal episode_count
        global epsilon
        global alpha
        if episode_count % 500 == 0:
            if args.episodes != -1:
                print(f"Episode: {episode_count}/{num_episodes}")
            else:
                print(f"Episode: {episode_count}")
        episode_count += 1

        command = SET_STATE
        if not args.run:
            if episode_count < 200:
                epsilon = (-0.45)*math.tanh(0.03*episode_count + -2.4) + 0.55
                print("epsilon                                                               ", epsilon)
                alpha = epsilon
            elif episode_count == 200:
                alpha = 0.2

            request_bytes = struct.pack('iffff', command, INIT_X , INIT_XDOT, random.randrange(200)/1000.0 -0.1 , INIT_THETADOT) # random.randrange(8)-4
        else:
            request_bytes = struct.pack('iffff', command, INIT_X, INIT_XDOT, INIT_THETA , INIT_THETADOT)

        socket.send(request_bytes)


    while args.episodes == -1 or episode_count <= num_episodes:    

        state, action = 0,0 # initilizing

        # Loop for each episode:
        # Initialize S
        # Loop for each step of episode:
        if count == 0:
            initState()
        elif LEFT_BOUND > x or x > RIGHT_BOUND:
            initState()
        elif theta < -1.6 or theta > 1.6:
            initState()
        else:
        #   Choose A from S using policy derived from Q (e.g., epsilon-greedy)
        #   Take action A, observe R, S`
            #greedy_policy_action = random_epsilon_greedy_policy(Q, epsilon, s, env.action_space.n) TODO
            #action = np.random.choice(np.arange(len(p_a)), p=p_a)
            state = translate_state(x, xdot, theta, thetadot) # translate location and angle into a state
            greedy_policy_action_probs = epsilon_greedy_policy(Q, state, len(actions))
            action = np.random.choice(len(actions), p=greedy_policy_action_probs) # choose action using greedy policy probabilities

            # print("theta", theta)
            # print("State", state)
            # print("Action", action)
            # print(episode_count, state,Q[state][:], action)

            # take a step
            command = APPLY_FORCE
            u = actions[action]
            request_bytes = struct.pack('if', command, u)
            socket.send(request_bytes)
        count += 1

        response_bytes = socket.recv()
        response_command, = struct.unpack('i', response_bytes[0:4])


        if response_command == NEW_STATE:
            x, xdot, theta, thetadot, reward = struct.unpack('fffff', response_bytes[4:])
            new_state = [x, xdot, theta, thetadot]
            if LEFT_BOUND > x or x > RIGHT_BOUND:
                reward = -10
        #   Q(S,A) <- Q(S,A) + alpha[R + gamma * max_a Q(S`, a) - Q(S,A)]
        #   S <- S`
            s_prime = translate_state(x, xdot, theta, thetadot)
            # print("state:", state, "action:", action)
            Q[state][action] = Q[state][action] + alpha * (reward + gamma * np.max(Q[s_prime][:]) - Q[state][action])
            # print("reward", reward)

        # if count % 100 == 0:
        #     print(Q)
        # until S is terminal

        if episode_count % 1000 == 0 and SAVE:
            # save every 100 episodes
            with open(outfile, 'wb') as f:
                np.save(f, Q)
            print("Q saved")
            print(Q)


    if SAVE:
        # save q values
        with open(outfile, 'wb') as f:
            np.save(f, Q)






def epsilon_greedy_policy(Q, s: int, num_actions: int):
    a = np.ones(num_actions, dtype=float) * epsilon/(num_actions-1)
    top_a = np.argmax(Q[s])
    a[top_a] = 1.0 - epsilon
    return a # example [0.6, 0.2, 0.2]



# translates x, xdot, theta, thetadot to a bucket
def translate_state(x: float, xdot: float, theta: float, thetadot: float) -> int:
    #dim4_index=x+y*x_arrsize+z*y_arrsize*x_arrsize+w*z_arrsize*y_arrsize*x_arrsize
    x_index = des(x, x_min, x_max, x_buckets)
    xdot_index = des(xdot, xdot_min, xdot_max, xdot_buckets)
    theta_index = des(theta, theta_min, theta_max, theta_buckets)
    thetadot_index = des(thetadot, thetadot_min, thetadot_max, thetadot_buckets)

    return x_index+xdot_index*x_buckets+theta_index*xdot_buckets*x_buckets+thetadot_index*theta_buckets*xdot_buckets*x_buckets


def des(float_val, min, max, num_buckets):
    if float_val > max:
        return num_buckets-1
    if float_val < min:
        float_val = 0
    width = abs(max-min)/num_buckets
    for i in range(0, num_buckets):
        if float_val < width * (i+1) + min:
            return i



if __name__ == "__main__":
    main()
