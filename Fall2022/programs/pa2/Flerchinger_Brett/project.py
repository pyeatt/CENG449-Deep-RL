#!/usr/bin/env python3

from scipy.integrate import solve_ivp
import inverted_pendulum as ip
import json
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import struct
import zmq
import random

def xConv(x):
    if x < -4.5:
        return 0
    elif x < -3.5:
        return 1
    elif x < -2.5:
        return 2
    elif x < -1.75:
        return 3
    elif x < -.75:
        return 4
    elif x < .75:
        return 5
    elif x < 1.75:
        return 6
    elif x < 2.5:
        return 7
    elif x < 3.5:
        return 8
    elif x < 4.5:
        return 9
    else:
        return 10

def xDotConv(xdot):
    if xdot < -20:
        return 0
    elif xdot < -13:
        return 1
    elif xdot < -8:
        return 2
    elif xdot < -3:
        return 3
    elif xdot < -.75:
        return 4
    elif xdot < .75:
        return 5
    elif xdot < 3:
        return 6
    elif  xdot < 8:
        return 7
    elif xdot < 13:
        return 8
    elif xdot < 20:
        return 9
    else:
        return 10

def thetaConv(theta):
    if theta < -np.pi/4:
        return 0
    elif theta < -np.pi/8:
        return 1
    elif theta < -np.pi/32:
        return 2
    elif theta < -np.pi/64:
        return 3
    elif theta < np.pi/64:
        return 4
    elif theta < np.pi/32:
        return 5
    elif theta < np.pi/8:
        return 6
    elif theta < np.pi/4:
        return 7
    elif theta < np.pi/2:
        return 8
    elif theta < np.pi:
        return 9
    else:
        return 10

def thetaDotConv(tdot):
    if tdot < -10:
        return 0
    elif tdot < -5.5:
        return 1
    elif tdot < -2.5:
        return 2
    elif tdot < -.75:
        return 3
    elif tdot < -.1:
        return 4
    elif tdot < .1:
        return 5
    elif tdot < .75:
        return 6
    elif tdot < 2.5:
        return 7
    elif tdot < 5.5:
        return 8
    elif tdot < 10:
        return 9
    else:
        return 10

def bAction(x1, x2):
    if x1 > x2:
        return 0
    else:
        return 1

def getReward(x, theta):
    if x > 5:
        return -50
    elif x < -5:
        return -50
    elif theta > np.pi/2:
        return -1
    elif theta < -np.pi/2:
        return -1
    else:
        return 0

def explore(rate):

    if random.randint(1, 100) < rate:
        return random.randint(0,1)
    else:
        return -1 



def main():
    APPLY_FORCE = 0
    SET_STATE = 1
    NEW_STATE = 2
    ANIMATE = 3

    values = np.zeros((11, 11, 11, 11, 2)) # x, xdot, theta, thetadot, action
    alpha = .55
    discount = .9
    exploreRate = 75 #percent 
    random.seed(7) # for replication purposes

    prevState = [xConv(-2), xDotConv(0), thetaConv(.2), thetaDotConv(0)]
    nextState = [xConv(-2), xDotConv(0), thetaConv(.2), thetaDotConv(0)]
    bestAction = 0
    action = 0 # using an explore rate, best action doesn't always match actual actoin.  Alsom used as previous action for update formula
    sumReward = 0

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    animation_enabled = True
    disabled = False
    count = 0
    while True:
        if count % 1000 == 0:
            print("Total Reward: ", sumReward)
            print("\nSimulation number: ", count/1000)
            
            sumReward = 0

            
            # toggle animation
            
            command = ANIMATE
            animation_enabled = not animation_enabled
            if count % 20000 == 0: # animate every 20th simulation
                request_bytes = struct.pack('ii', command, animation_enabled)
            else:
                 request_bytes = struct.pack('ii', command, disabled)
            socket.send(request_bytes)

            if exploreRate > 0:
                exploreRate -= 5 #decrease explore rate as more states have been encountered

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

        else:
            command = APPLY_FORCE

            action = explore(exploreRate)
            if action == -1:
                action = bestAction

            if action == 0:
                u = 200
            else:
                u = -200
            request_bytes = struct.pack('if', command, u)
            socket.send(request_bytes)

        count += 1

        response_bytes = socket.recv()
        response_command, = struct.unpack('i', response_bytes[0:4])

        if response_command == NEW_STATE:
            x, xdot, theta, thetadot, reward = struct.unpack(
                'fffff', response_bytes[4:])
            

            prevState = nextState
            nextState = [xConv(x), xDotConv(xdot), thetaConv(theta), thetaDotConv(thetadot)]
            myReward = getReward(x, theta)
            sumReward += myReward

            bestAction = bAction(values[nextState[0]][nextState[1]][nextState[2]][nextState[3]][0], values[nextState[0]][nextState[1]][nextState[2]][nextState[3]][1])

            if x > 5:
                count += 1000 - count%1000 # episode ends
                print("This simulation goes OOB")

            if x < -5:
                count += 1000 - count%1000 # episode ends
                print("This simulation goes OOB")

            # I know the update formula is pretty ugly.  Usually use c++ and forcing myself to learn python: plan on making codes neater in the future.
            values[prevState[0]] [prevState[1]] [prevState[2]] [prevState[3]] [action] += alpha*(myReward + discount * values[nextState[0]][nextState[1]][nextState[2]][nextState[3]][bestAction]  - values[prevState[0]] [prevState[1]] [prevState[2]] [prevState[3]] [action])

            
        elif response_command == ANIMATE:
            animation_enabled, = struct.unpack('i', response_bytes[4:])
        else:
            print("Error: invalid command: ", response_command)


if __name__ == "__main__":
    main()
