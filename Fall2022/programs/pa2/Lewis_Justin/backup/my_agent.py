#!/usr/bin/env python3

import numpy as np
import struct
import zmq
import random
import pickle
import sys
import traceback

FORCE = 3
ACTIONS = [-FORCE, 0, FORCE] # left, nothing, right

X_IDX = 0
V_IDX = 1
THETA_IDX = 2
A_V_IDX = 3 # angular velocity
REWARD_IDX = 4 # only for the response from server

X_BOUNDS = (-5, 5)
X_DOT_BOUNDS = (-1000, 1000) # no bounds!
THETA_BOUNDS = (-0.8, 0.8)
THETA_DOT_BOUNDS = (-1000, 1000) # no bounds! 

# EPISODES_PER_TRAIN = 500  # Only guarunteed convergence to optimal policy if inifinity
                          #  This might as well be, since it takes ages to go through an episode
STEPS_PER_EPISODE = 900

APPLY_FORCE = 0
SET_STATE = 1
NEW_STATE = 2
ANIMATE = 3

def discretizeState(contState):
    x = 1  # I don't want to worry about x position, I set server to just wrap around
           # yes I could make the state space larger, but that just adds training time when
           # it doesn't contribute to educational value
    v = np.sign(contState[V_IDX])
    a = round(contState[THETA_IDX], 1) # angle to 1 decimal point
    av = np.sign(contState[A_V_IDX]) # angular velocity 
    # RE theta, top is 0, left is positive, right is negative. horizontal is pi/2, and down is pi (or -pi)
    return (x, v, a, av)

def eGreedy(q, s, epsilon):
    if (random.random() <= epsilon):
        return random.choice(ACTIONS)
    # otherwise, just take the assumed best action for the current state
    best = None
    bestValue = None
    s = discretizeState(s)
    for action in ACTIONS:
        currValue = q.get((s, action), 0)
        if bestValue is None or bestValue > currValue:
            bestValue = currValue
            best = action

    if best is None:
        print("Scream and die")
        exit(-1)
    return best


# THESE TWO HAVE BEEN REMOVED BECAUSE I REMEMBERED OBJECTS CAN HAVE A DEFAULT VALUE IF YOU USE .get()
# def checkQ(Q):
    # nQ = initQ()
    # for key in nQ.keys():
        # if key not in Q.keys():
            # Q[key] = 0


# def initQ():  # not necessary but c'est la vie
    # # I'm fairly confident that this could be done only with velocity, 
    # # since the agent should just move in a way that cancels out the velocity. 
    # # but I'm going to try to do it with more for practice.
# 
    # # OUTDATED # x space from [-5, 5]
    # # OUTDATED # so there should be 11 buckets
    # xVals = [1]
# 
    # # I think velocity could be simply negative, 0, or positive
    # vVals = [-1, 0, 1]
# 
    # aVals = np.linspace(-np.pi/2, np.pi/2, 17)
    # aVVals = [-1, 0, 1]
# 
    # # theta will be ignored since there seems to be a bug
    # q = {}
    # for x in xVals:
        # for v in vVals:
            # for a in aVals:
                # for aV in aVVals:
                    # for action in ACTIONS:
                        # a = round(a, 1)
                        # state = (x, v, a, aV)
                        # q[state, action] = 0
    # return q

def initS(socket):
    x = 0
    xdot = 0.0
    theta = random.choice(np.linspace(THETA_BOUNDS[0]/2, THETA_BOUNDS[1]/2, 15)) # random starting angle
    # theta = 0.2
    thetadot = 0.0

    request_bytes = struct.pack(
        'iffff', SET_STATE, x, xdot, theta, thetadot)
    socket.send(request_bytes)
    response_bytes = socket.recv()
    response_command, = struct.unpack('i', response_bytes[0:4])
    # returns a new state but don't care to do anything with it \o/
    # if response_command == NEW_STATE:
        # x, xdot, theta, thetadot, reward = struct.unpack(
            # 'fffff', response_bytes[4:])
        # new_state = [x, xdot, theta, thetadot]
    # else:
        # print("Error: invalid command: ", response_command)
        # raise Exception("Error initS")
    return (x, xdot, theta, thetadot)


def takeAction(action, socket):
    request_bytes = struct.pack('if', APPLY_FORCE, action)
    socket.send(request_bytes)
    response_bytes = socket.recv()
    response_command, = struct.unpack('i', response_bytes[0:4])
    

    if response_command == NEW_STATE:
        x, xdot, theta, thetadot, reward = struct.unpack(
            'fffff', response_bytes[4:])
        new_state = (x, xdot, theta, thetadot)
        if x < X_BOUNDS[0] or x > X_BOUNDS[1] or \
                xdot < X_DOT_BOUNDS[0] or xdot > X_DOT_BOUNDS[1] or \
                theta < THETA_BOUNDS[0] or theta > THETA_BOUNDS[1] or \
                thetadot < THETA_DOT_BOUNDS[0] or thetadot > THETA_DOT_BOUNDS[1]:
            reward = -1

        return new_state, reward
    else:
        print("Error: invalid command: ", response_command)
        raise Exception("IDK What to do with this")

def SetAnimate(animate, socket):
    request_bytes = struct.pack('ii', ANIMATE, animate)
    socket.send(request_bytes)

    response_bytes = socket.recv()
    response_command, = struct.unpack('i', response_bytes[0:4])

    if response_command == ANIMATE:
        # I don't really know what to do with this, but it should only be sent
        # when I send it an animate command so I'm sure it's fine
        struct.unpack('i', response_bytes[4:]) # Probably not necessary
                                               # I don't know much about struct though, not sure
                                               # if queue that I need to keep clear. 
    else:
        print("Error: invalid command: ", response_command)
        raise Exception("IDK What to do with this")


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    Q = {}
    if "-l" in sys.argv or "--load" in sys.argv:
        print("loading")
        with open('qVals.obj', 'rb') as fin:
            Q =  pickle.load(fin)
    try:
        SetAnimate(False, socket)
        '''  SARSA Algorithm
        Algorithm parameters: step size alpha \in (0, 1], small e > 0               # I'm not sure what this should be so I'll say alpha = 1, can be adjusted easily though
        Initialize Q(s, a), for all s \in S+, a \in A(s), 
                    arbitrarily except that Q(terminal, ·) = 0

        Loop for each episode:                                                      # episode -> reset
            Initialize S                                                            # e.g. the section with command = SET_STATE from example_agent.py
            Choose A from S using policy derived from Q (e.g., e-greedy)            # 
            Loop for each step of episode:                                          # steps in an episode don't really exist, since it's continuous, but every send & response iteration
                Take action A, observe R, S'                                        # \o/ send the packet, get the response
                Choose A' from S' using policy derived from Q (e.g., e-greedy)
                Q(S, A) <- Q(S, A) + a[R + gamma*Q(S', A') - Q(S, A)]

                S <- S'
                A <- A'
            until S is terminal                                                     # There isn't really a terminal state, so we'll need to decide when to reset an episode
                                                                                    # We can probably call the algorithm done if we can do, say, 3 consecutive episodes without
                                                                                    #   dropping the pole
        '''
        epsilon = 0.03   #
        alpha = .3        # ????, technically .001 I think from the server timestep
        gamma = 1
        iteration = 0

        if "-a" in sys.argv or "--animate" in sys.argv:
            SetAnimate(True, socket)
        while True:
            iteration += 1
            if iteration % 1000 == 0:
                print(iteration)

            S = initS(socket)
            S = discretizeState(S)
            A = eGreedy(Q, S, epsilon)
            for _ in range(STEPS_PER_EPISODE):
                Sp, r = takeAction(A, socket)
                Sp = discretizeState(Sp)
                Ap = eGreedy(Q, Sp, epsilon)
                Q[S, A] = Q.get((S, A), 0) + alpha * (r + gamma*Q.get((Sp, Ap), 0) - Q.get((S, A), 0))
                S = Sp
                A = Ap
                # print(">" if A > 0 else "<" if A < 0 else "o", end='\r', flush=True)
                if r < 0: # end of episode
                    break
    except KeyboardInterrupt as e:
        if "-s" in sys.argv or "--save" in sys.argv:
            print("dumping")
            with open('qVals.obj', 'wb') as fout:
                pickle.dump(Q, fout)
    except Exception as e:
        socket.close()
        print(str(e))
        print(traceback.format_exc())



if __name__ == "__main__":
    main()
