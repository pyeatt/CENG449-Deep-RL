# Noah Zikmund
# CENG 449
# Dr. Pyeatt
# 19 Sept 2022

# PA1: Gridworld Problem
#---------------------------------------------------------------------------------------------

#       0       1       2       3
#
#       4       5       6       7
#
#       8       9       10      11
#
#       12      13      14      15

#---------------------------------------------------------------------------------------------

# 8 LEFT -> 15                                                 reward = -2     probability = 1
# 15 RIGHT or DOWN -> 15                                       reward = 0      probability = 1
# Any move off of the grid -> back to your original space      reward = -1     probability = 1
# Any move to an adjacent number -> adjacent number            reward = -1     probability = 1

# Global State-Value Function: These arrays hold the values of each state. 
# stateValue holds the state values
stateValue = [0]*16
policyValue = ['0']*16

# policy is a 16 x 4 (row x col) array that holds the probability that each state action pair has of occuring
# Ex. policy[0][1] is the probability at which state 0 and action 1 occurs (0, D) 
# For the first run-through, every move has a 1/4 probability
policy = [0.25]*64
num_policy = 0

actions = ['U', 'D', 'L', 'R']
states = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', "10", "11", "12", "13", "14", "15"]

# calculateNextState is given a state-action pair and returns the resulting state
def calculateNextState(state, action):

    if(state == '0'):
        if(action == 'U'):
            nextState = '0'
        elif(action == 'D'):
            nextState = '4'
        elif(action == 'L'):
            nextState = '0'
        else:
            nextState = '1'

    elif(state == '1'):
        if(action == 'U'):
            nextState = '1'
        elif(action == 'D'):
            nextState = '5'
        elif(action == 'L'):
            nextState = '0'
        else:
            nextState = '2'

    elif(state == '2'):
        if(action == 'U'):
            nextState = '2'
        elif(action == 'D'):
            nextState = '6'
        elif(action == 'L'):
            nextState = '1'
        else:
            nextState = '3'

    elif(state == '3'):
        if(action == 'U'):
            nextState = '3'
        elif(action == 'D'):
            nextState = '7'
        elif(action == 'L'):
            nextState = '2'
        else:
            nextState = '3'

    elif(state == '4'):
        if(action == 'U'):
            nextState = '0'
        elif(action == 'D'):
            nextState = '8'
        elif(action == 'L'):
            nextState = '4'
        else:
            nextState = '5'

    elif(state == '5'):
        if(action == 'U'):
            nextState = '1'
        elif(action == 'D'):
            nextState = '9'
        elif(action == 'L'):
            nextState = '4'
        else:
            nextState = '6'

    elif(state == '6'):
        if(action == 'U'):
            nextState = '2'
        elif(action == 'D'):
            nextState = '10'
        elif(action == 'L'):
            nextState = '5'
        else:
            nextState = '7'

    elif(state == '7'):
        if(action == 'U'):
            nextState = '3'
        elif(action == 'D'):
            nextState = '11'
        elif(action == 'L'):
            nextState = '6'
        else:
            nextState = '7'

    elif(state == '8'):
        if(action == 'U'):
            nextState = '4'
        elif(action == 'D'):
            nextState = '12'
        elif(action == 'L'):
            nextState = '15'
        else:
            nextState = '9'

    elif(state == '9'):
        if(action == 'U'):
            nextState = '5'
        elif(action == 'D'):
            nextState = '13'
        elif(action == 'L'):
            nextState = '8'
        else:
            nextState = '10'

    elif(state == '10'):
        if(action == 'U'):
            nextState = '6'
        elif(action == 'D'):
            nextState = '14'
        elif(action == 'L'):
            nextState = '9'
        else:
            nextState = '11'

    elif(state == '11'):
        if(action == 'U'):
            nextState = '7'
        elif(action == 'D'):
            nextState = '15'
        elif(action == 'L'):
            nextState = '10'
        else:
            nextState = '11'

    elif(state == '12'):
        if(action == 'U'):
            nextState = '8'
        elif(action == 'D'):
            nextState = '12'
        elif(action == 'L'):
            nextState = '12'
        else:
            nextState = '13'
 
    elif(state == '13'):
        if(action == 'U'):
            nextState = '9'
        elif(action == 'D'):
            nextState = '13'
        elif(action == 'L'):
            nextState = '12'
        else:
            nextState = '14'

    elif(state == '14'):
        if(action == 'U'):
            nextState = '10'
        elif(action == 'D'):
            nextState = '14'
        elif(action == 'L'):
            nextState = '13'
        else:
            nextState = '15'

    # State 15
    else:
        if(action == 'U'):
            nextState = '11'
        elif(action == 'D'):
            nextState = '15'
        elif(action == 'L'):
            nextState = '14'
        else: 
            nextState = '15'
    
    return nextState

# calculateReward is given a state-action pair and returns the resulting reward
def calculateReward(state, action):
    
    reward = -1

    if(state == '15' and (action == 'D' or action == 'R')):
        reward = 0
    elif(state == '8' and action == 'L'):
        reward = -2
    else:
        reward = -1
    
    return reward

# bellmanEq is given a probability (from the policy) from a known state-action pair, a reward
# from a known state-action pair, and the next state from a known state-action pair which
# returns the state-value for a known state
def bellmanEq(probability, reward, nextState):

    global stateValue

    retVal = probability * (reward + ((0.95) * stateValue[int(nextState)]))

    return retVal

# evaluate populates the array stateValue with each state's state-value
def evaluate():
    global actions, policy, states, stateValue
    row = 0
    probability = 0
    value = 0
    action = ' '

    if(num_policy == 0):
        while(1):
            maxChange = 0
            for i in range(16):
                value = 0
                for j in range(4):
                    action = actions[j]
                    if(action == 'U'):
                        row = 0
                    elif(action == 'D'):
                        row = 1
                    elif(action == 'L'):
                        row = 2
                    else:
                        row = 3

                    probability = policy[row + i * 4]
                    nextState = calculateNextState(states[i], action)
                    reward = calculateReward(states[i], action)
                    value += (bellmanEq(probability, reward, nextState))
                        
                if((abs(abs(stateValue[i]) - abs(value))) > maxChange):
                    maxChange = abs(abs(stateValue[i]) - abs(value)) 
                stateValue[i] = value
            
            if(maxChange < 0.001):
                return

    elif(num_policy == 1):
        for i in range(16):  
            greedyPolicy(i)
            for j in range(4):
                if(policy[j + 1 * 4] == 1):
                    action = actions[j]

            nextState = calculateNextState(states[i], action)
            reward = calculateReward(states[i], action)
            stateValue[i] = (bellmanEq(1, reward, nextState))

# formatStateValue is a print function that formats the results neatly
def formatStateValue():

    global stateValue

    increment = 0

    print("\n")

    for x in range(4):
        print(" {:<6} {:<6} {:<6} {:<6} ".format('{:.2f}'.format(stateValue[increment]), '{:.2f}'.format(stateValue[increment + 1]),
                '{:.2f}'.format(stateValue[increment + 2]), '{:.2f}'.format(stateValue[increment + 3])))
        print("\n")
        increment += 4

def greedyPolicy(state):
    global stateValue, policy, states

    maximum = 0

    up = calculateNextState(states[(state)], 'U')
    down = calculateNextState(states[state], 'D')
    left = calculateNextState(states[state], 'L')
    right = calculateNextState(states[state], 'R')

    maximum = stateValue[int(up)]
    policy[0 + int(state) * 4] = 1
    policy[1 + int(state) * 4] = 0
    policy[2 + int(state) * 4] = 0
    policy[3 + int(state) * 4] = 0

    if(stateValue[int(down)] > maximum):
        maximum = stateValue[int(down)]
        policy[0 + int(state) * 4] = 0
        policy[1 + int(state) * 4] = 1
        policy[2 + int(state) * 4] = 0
        policy[3 + int(state) * 4] = 0

    if(stateValue[int(left)] > maximum):
        maximum = stateValue[int(left)]
        policy[0 + int(state) * 4] = 0
        policy[1 + int(state) * 4] = 0
        policy[2 + int(state) * 4] = 1
        policy[3 + int(state) * 4] = 0

    if(stateValue[int(right)] > maximum):
        maximum = stateValue[int(right)]
        policy[0 + int(state) * 4] = 0
        policy[1 + int(state) * 4] = 0
        policy[2 + int(state) * 4] = 0
        policy[3 + int(state) * 4] = 1   

def printPolicy():
    global policyValue, policy
    print("\n")
    printArr = ["Up", "Down", "Left", "Right"]
    for x in range(16):
        for y in range(4):
            if(policy[y + x * 4] == 1):
                policyValue[x] = printArr[y]
    
    increment = 0

    for x in range(4):
        print(" {:<6} {:<6} {:<6} {:<6} ".format(policyValue[increment], policyValue[increment + 1], policyValue[increment + 2], policyValue[increment + 3]))
        print("\n")
        increment += 4

def main():
    
    global stateValue, num_policy
    increment = 0
    
    # First run with random policy
    print("-----------------------------------")
    print("Value Function with Random Policy:")
    evaluate()
    formatStateValue()

    num_policy += 1
    evaluate()

    num_policy = 0
    # Second run with improved (greedy) policy
    print("-----------------------------------")
    print("Value Function with Improved Policy")
    evaluate()
    formatStateValue()

    print("----------------------------------")
    print("Deterministic Policy")
    printPolicy()

if __name__ == "__main__":
    main()