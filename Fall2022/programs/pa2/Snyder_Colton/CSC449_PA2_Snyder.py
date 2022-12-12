# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:38:51 2022

@author: Colton Snyder
"""

from enum import Enum, IntEnum
import numpy as np
import zmq
import struct
import random
from math import pi, inf
import pickle
import sys

class Commands(IntEnum):
    APPLY_FORCE = 0
    SET_STATE = 1
    NEW_STATE = 2
    ANIMATE = 3

class Actions(Enum):
    LargeLeft = -10
    MidLeft = -5
    ShortLeft = -1
    Stay = 0
    ShortRight = 1
    MidRight = 5
    LargeRight = 10
    
def getStateReward(x, theta):
    if abs(theta) >= pi/2 or abs(x) >= 7:
        return -100
    elif abs(theta) >= pi/4:
        return -1
    else:
        return 0


def main(fileName):
    # Connect to Server
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    
    # Define hyperparameters
    STEP_SIZE = 0.9
    EPSILON = 0.03
    GAMMA = 0.5
    
    XStateRange = [-5,5]
    XStatePartitions = 10
    XDotStateRange = [-5,5]
    XDotStatePartitions = 10
    ThetaStateRange = [-pi/2,pi/2]
    ThetaStatePartitions = 10
    ThetaDotStateRange = [-pi/2,pi/2]
    ThetaDotStatePartitions = 10
    
    episodeNum = 0
    
    if fileName:
        file = open(fileName, 'rb')
        StateActionValue = pickle.load(file)
        file.close()
        
        episodeNum = int(fileName.split('_')[1].split('.')[0]) * 100
    else:
        StateActionValue = np.zeros((XStatePartitions, XDotStatePartitions,
                                 ThetaStatePartitions, ThetaDotStatePartitions, len(list(Actions))))
    
    
    
    while True:
        # Initialize State
        command = Commands.SET_STATE
        x = random.random() * 4 - 2
        xdot = random.random() * 1 - 0.5
        theta = random.random() * 0.5 - 0.25
        thetadot = 0
        request_bytes = struct.pack(
            'iffff', command, x, xdot, theta, thetadot)
        socket.send(request_bytes)
        
        response_bytes = socket.recv()
        response_command, = struct.unpack('i', response_bytes[0:4])
        
        episodeLength = 0
        
        while getStateReward(x, theta) > -100:
            episodeLength += 1
            
            x_ind = (x - XStateRange[0]) * XStatePartitions // (XStateRange[1] - XStateRange[0])
            x_ind = int(max(0, min(XStatePartitions - 1, x_ind)))
            xdot_ind = (xdot - XDotStateRange[0]) * XDotStatePartitions // (XDotStateRange[1] - XDotStateRange[0])
            xdot_ind = int(max(0, min(XDotStatePartitions - 1, xdot_ind)))
            theta_ind = (theta - ThetaStateRange[0]) * ThetaStatePartitions // (ThetaStateRange[1] - ThetaStateRange[0])
            theta_ind = int(max(0, min(ThetaStatePartitions - 1, theta_ind)))
            thetadot_ind = (thetadot - ThetaDotStateRange[0]) * ThetaDotStatePartitions // (ThetaDotStateRange[1] - ThetaDotStateRange[0])
            thetadot_ind = int(max(0, min(ThetaDotStatePartitions - 1, thetadot_ind)))
            
            # Pick Next Action from current state using e-greedy
            next_act = Actions.Stay
            act_ind = 2
            if random.random() > EPSILON:
                max_val = -inf
                for i, act in enumerate(Actions):
                    # print(x_ind, xdot_ind, theta_ind, thetadot_ind, i)
                    val = StateActionValue[x_ind, xdot_ind, theta_ind, thetadot_ind, i]
                    if val > max_val:
                        max_val = val
                        next_act = act
                        act_ind = i
            else:
                act_ind = random.randint(0, len(list(Actions))-1)
                next_act = list(Actions)[act_ind]
                
            command = Commands.APPLY_FORCE
            request_bytes = struct.pack('if', command, next_act.value)
            socket.send(request_bytes)
            
            response_bytes = socket.recv()
            response_command, = struct.unpack('i', response_bytes[0:4])
            
            if response_command == Commands.NEW_STATE:
                x, xdot, theta, thetadot, reward = struct.unpack(
                    'fffff', response_bytes[4:])
                new_state = [x, xdot, theta, thetadot]
                reward = getStateReward(x, theta)
                currVal = StateActionValue[x_ind, xdot_ind, theta_ind, thetadot_ind, act_ind]
                
                newx_ind = (x - XStateRange[0]) * XStatePartitions // (XStateRange[1] - XStateRange[0])
                newx_ind = int(max(0, min(XStatePartitions - 1, newx_ind)))
                newxdot_ind = (xdot - XDotStateRange[0]) * XDotStatePartitions // (XDotStateRange[1] - XDotStateRange[0])
                newxdot_ind = int(max(0, min(XDotStatePartitions - 1, newxdot_ind)))
                newtheta_ind = (theta - ThetaStateRange[0]) * ThetaStatePartitions // (ThetaStateRange[1] - ThetaStateRange[0])
                newtheta_ind = int(max(0, min(ThetaStatePartitions - 1, newtheta_ind)))
                newthetadot_ind = (thetadot - ThetaDotStateRange[0]) * ThetaDotStatePartitions // (ThetaDotStateRange[1] - ThetaDotStateRange[0])
                newthetadot_ind = int(max(0, min(ThetaDotStatePartitions - 1, newthetadot_ind)))
                
                max_val = -inf
                for i, act in enumerate(Actions):
                    val = StateActionValue[newx_ind, newxdot_ind, newtheta_ind, newthetadot_ind, i]
                    if val > max_val:
                        max_val = val
                
                StateActionValue[x_ind, xdot_ind, theta_ind, thetadot_ind, act_ind] += STEP_SIZE * (reward + GAMMA * max_val - currVal)
                # print(f"{x_ind=}, {xdot_ind=}, {theta_ind=}, {thetadot_ind=}, {i=}", reward, STEP_SIZE * (reward + GAMMA * max_val - currVal))
            else:
                print("Error: invalid command: ", response_command)
        
        print(f"Episode {episodeNum} ended, {episodeLength=}")
        if episodeNum % 100 == 0:
            file = open(f"StateActionValue_{episodeNum//100}.txt", 'wb')
            pickle.dump(StateActionValue, file)
            file.close()
        episodeNum += 1
    
    
if __name__ == "__main__":
    fileName = sys.argv[1] if len(sys.argv) > 1 else None
    main(fileName)