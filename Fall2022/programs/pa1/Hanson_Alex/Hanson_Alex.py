# Alex Hanson
# CSC 449
# Programming Assignment 1
# Python version 3.9.0
#
# I am Undergraduate student but I did do the stochastic policy as well just because I wanted too.
#
#
#
# Write a program that calculates and prints out one optimal deterministic policy for this variation of the
# Gridworld, and the value function v∗(s) for that policy.



numStates = 16
numActions = 4
gamma = 0.99
actions = [0,1,2,3] # up, down, left, right
states = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
V = [0.0]*16 # state values
#pi = [ [0.25]*4 ]*16
pi = [[0.25]*4 for x in range(16)]


nState = [
    [0, 4, 0, 1],
    [1, 5, 0, 2],
    [2, 6, 1, 3],
    [3, 7, 2, 3],
    [0, 8, 4, 5],
    [1, 9, 4, 6],
    [2, 10, 5, 7],
    [3, 11, 6, 7],
    [4, 12, 15, 9], # magic teleporter
    [5, 13, 8, 10],
    [6, 14, 9, 11],
    [7, 15, 10, 11],
    [8, 12, 12, 13],
    [9, 13, 12, 14],
    [10, 14, 13, 15],
    [11, 15, 14, 15] # END state
]







def printV():
    print("Value")
    for i in range(numStates):
        print(" {:.8f} ".format(V[i]), end='')
        if( (i+1) % 4) == 0:
            print()
    print()


def printPi():
    print("Policy")
    print("State |UP  |DOWN |LEFT |RIGHT")
    for i in range(numStates):
        print('{:5}'.format(i), end=' ')
        for j in range(numActions):
            print(" {:4.2f} ".format(pi[i][j]), end='')
        print()
    print()







# returns next state that results from inital state s taking action a
# deterministic so 100% chance of going to next state
def nextState(s, a):
    return nState[s][a]

# returns reward for transition from state s to next state ns taking action a
def reward(s, a, ns):
    if s == 8 and a == 2 and ns == 15: # magic teleporter
        return -2
    if ns == 15 and (a == 3 or a == 1): # terminal state
        return 0
    return -1






# evaluates 1 ideration of the current policy pi
# and update value of states V
def policyEvaluation():
    global V
    newV = [0] * numStates
    for s in states:
        v = 0
        for a in actions:
            prob = pi[s][a] # probablity of taking action a in state s under current policy
            ns = nextState(s, a) # get next state if action a is taken
            rs = reward(s, a, ns) # reward for taking action a
            v += prob * (rs + gamma * V[ns]) # bellman
        newV[s] = v
    V = newV

            

# updates the current policy pi using new information from value function
def policyIteration():
    for s in states: # for each state
        maxV = V[s]
        count = 0
        value = [0]*4
        for a in actions: # for each action
            ns = nextState(s, a) # next state if action a is taken
            rs = reward(s, a, ns) # reward for state
            v = rs + gamma * V[ns] # value function
            #save for updating actions
            value[a] = v
            #update max
            if a == 0 or v > maxV:
                maxV = v
                count = 1
            elif v == maxV:
                count+=1
        # update policy, average over actions
        for a in actions:
            v = value[a]
            if v == maxV:
                pi[s][a] = 1.0 / count # averaging over the actions
                # using count because then if 2 actions have same value
                # it results in a 0.5,0.5 split between them
            else:
                pi[s][a] = 0
            


# updates the current policy pi using new information from value function
def policyIterationDeterministic():
    for s in states: # for each state
        maxV = 0
        value = [0]*4
        for a in actions: # for each action
            ns = nextState(s, a) # next state if action a is taken
            rs = reward(s, a, ns) # reward for taking action a in state s
            v = rs + gamma * V[ns] # value function
            #save for update
            value[a] = v
            #update max
            if a == 0 or v > value[maxV]:
                maxV = a
                
        # update policy, set new action
        for a in actions:
            if a == maxV:
                pi[s][a] = 1.0 # deterministic
            else:
                pi[s][a] = 0.0
















print("=================Deterministic=================")
print("Initial")
printV()
print("Initial")
printPi()
# episodes
for i in range(1000):
    # policy evaluation
    policyEvaluation()
    
    # policy iteration
    #policyIteration()
    policyIterationDeterministic()
printV()
printPi()





print("=================Stochastic=================")
pi = [[0.25]*4 for x in range(16)] # reseting pi for stochastic
V = [0.0]*16 # resetting for stochastic

print("Initial")
printV()
print("Initial")
printPi()
# episodes
for i in range(1000):
    # policy evaluation
    policyEvaluation()
    
    # policy iteration
    policyIteration()
printV()
printPi()













