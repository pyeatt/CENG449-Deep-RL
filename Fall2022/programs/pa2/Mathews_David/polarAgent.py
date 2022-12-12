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
import sys
from bisect import bisect_left
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def main():
    APPLY_FORCE = 0
    SET_STATE = 1
    NEW_STATE = 2
    ANIMATE = 3

    LEFT = 0
    STAY = 1
    RIGHT = 2
    ACTIONS = [-70, 0, 70]
    choice = STAY
    epsilon = 0#0.05
    printmark = False
    alpha = 0.9
    discount = 0.99
    
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    xnum, xdotnum, thetanum, thetadotnum = 12, 6, 20, 6
    xbounds = np.linspace(-3,3, xnum)
    xdotbounds = np.linspace(-2,2,xdotnum)
    thetabounds = np.linspace(-np.pi/2,np.pi/2,thetanum)
    thetadotbounds = np.linspace(-12,12,thetadotnum)

    animation_enabled = False
    output_file = False
    output_string = "PolarTraining"
    anCount = 0
    epLength = 1
    count = 0
    choice_arr = []
    state = [0,0,0,0]
    new_state = [0,0,0,0]

    if len(sys.argv) > 1:
        with open(str(sys.argv[1]), "r") as read_file:
            inputData = json.load(read_file)
            value = np.asarray(inputData["Qfunc"])
            anCount = int(sys.argv[2])
    else:
        value = np.zeros(shape = (xnum, xdotnum, thetanum, thetadotnum, len(ACTIONS)))
        

    while True:
        if count % 1000 == 0:

            # toggle animation
            command = ANIMATE
            if anCount % epLength == 0:
                 animation_enabled = True
                 if anCount % 10000 == 0:
                    output_file = True
                 else:
                    output_file = False
            else:
                 animation_enabled = False
            request_bytes = struct.pack('ii', command, animation_enabled)
            socket.send(request_bytes)
        
        elif count % 1000 == 1:
            # reset the state
            command = SET_STATE
            x = 0.0
            xdot = 0.0
            theta = 0.1
            thetadot = 0.0
            printmark = False
            request_bytes = struct.pack(
                'iffff', command, x, xdot, theta, thetadot)
            socket.send(request_bytes)
            resetSim = False

        else:
            command = APPLY_FORCE
            choice = get_move(value, new_state, epsilon)
            u = ACTIONS[choice]
            #out = value[int(state[0]),int(state[1]),int(state[2]),int(state[3]),int(choice)]
            #print(out,u)
            request_bytes = struct.pack('if', command, u)
            socket.send(request_bytes)

        count += 1

        response_bytes = socket.recv()
        response_command, = struct.unpack('i', response_bytes[0:4])

        if response_command == NEW_STATE:
            #store state
            state[0] = take_closest(xbounds,x)
            state[1] = take_closest(xdotbounds,xdot)
            state[2] = take_closest(thetabounds,theta)
            state[3] = take_closest(thetadotbounds,thetadot)
            #state = [x,xdot,theta,thetadot]
            #get new state
            x, xdot, theta, thetadot, reward = struct.unpack(
                'fffff', response_bytes[4:])
            #round based on bounds
            new_state[0] = take_closest(xbounds,x)
            new_state[1] = take_closest(xdotbounds,xdot)
            new_state[2] = take_closest(thetabounds,theta)
            new_state[3] = take_closest(thetadotbounds,thetadot)
            #new_state = [x, xdot, theta, thetadot]
            #Reset animation if out of bounds
            if reward == -1 or count > 10000:
                #printmark = True
                anCount += 1
                print (anCount)
                #print (theta, thetadot)
                #print (choice_arr)
                #print(stateVal,MoveValue,choice, max_choice(value, new_state))
                if (animation_enabled == True and output_file == True) or count > 10000:
                    outputData = {"Qfunc": value}
                    with open("./OutFiles/"+output_string+str(anCount)+".json", "w") as write_file:
                        json.dump(outputData, write_file, cls=NumpyArrayEncoder)
                    animation_enabled = True
                count = 0
                #choice_arr = []
            #Calc TD
            stateVal = value[int(state[0]),int(state[1]),int(state[2]),int(state[3]),int(choice)]
            MoveValue = alpha * (reward + discount * (max_choice(value, new_state) - stateVal))
            value[int(state[0]),int(state[1]),int(state[2]),int(state[3]),int(choice)] += MoveValue 
            #choice_arr.append(choice)

            if printmark == True:
                print (state[0],state[1],state[2],state[3],choice, max_choice(value,state),MoveValue)
                printmark = False
            #print(new_state, reward)
        elif response_command == ANIMATE:
            animation_enabled, = struct.unpack('i', response_bytes[4:])
        else:
            print("Error: invalid command: ", response_command)

def take_closest(myList, myNumber):
    pos = 0
    for i in range(len(myList)):
        val = abs(myList[i] - myNumber);
        if abs(myList[pos] - myNumber) < val:
            pos = i
    return pos
    
def max_choice(value, state):
    choice = 0
    numactions = np.shape(value)
    for i in range(numactions[4]):
        if value[int(state[0]),int(state[1]),int(state[2]),int(state[3]),i] > value[int(state[0]),int(state[1]),int(state[2]),int(state[3]),choice]:
            choice = i
    return value[int(state[0]),int(state[1]),int(state[2]),int(state[3]),choice]

def get_move(value, state, epsilon):
    choice = 0
    numactions = np.shape(value)
    if epsilon * 100 >= random.randint(1,100):
        choice = random.randint(0,numactions[4]-1)
        #print ("RandMove")
        return choice
    for i in range(numactions[4]):
        if value[int(state[0]),int(state[1]),int(state[2]),int(state[3]),i] > value[int(state[0]),int(state[1]),int(state[2]),int(state[3]),choice]:
            choice = i
    #print (choice)
    return choice

if __name__ == "__main__":
    main()
