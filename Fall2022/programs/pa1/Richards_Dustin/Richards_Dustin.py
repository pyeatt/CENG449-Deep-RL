# known working under Python 3.9.2

from enum import Enum
from copy import deepcopy
import random
import numpy as np
import pdb
# ===== CONSTANTS =====
numStates = 16
gamma = 0.95 # discount factor
theta = 0.001 # accuracy of estimation

class Action(Enum):
    left = 0
    down = 1
    right = 2
    up = 3

# stochastic action, contains one or more Actions which will be randomly chosen from
class StocAction():
    def __init__(self, actions : list[Action] = []):
        self.actions = actions

    def __repr__(self):
        s = "| "
        for action in self.actions:
            if (action == Action.left):
                s += "<"
            elif (action == Action.up):
                s += "^"
            elif (action == Action.down):
                s += "v"
            elif (action == Action.right):
                s += ">"
        s += " " * (4 - len(self.actions))
        s += " |"

        return s

    def getAction(self) -> Action:
        return random.choose(self.actions)

def reward(state: int, action: Action) -> float:
    # magic teleporter
    if (state == 8 and action == Action.left):
        return -2
    # goal state
    elif (state == 15 and (action == Action.down or action == Action.right)):
        return 0
    # everything else
    else:
        return -1

# ===== BORDER DETECT FUNCTIONS =====
def isTopBorder(state: int) -> bool:
    return (state >= 0 and state <= 3)

def isBottomBorder(state: int) -> bool:
    return (state >= 12 and state <= 15)

def isLeftBorder(state: int) -> bool:
    if (state == 8):
        return False

    return (state == 0 or state == 4 or state == 12)

def isRightBorder(state: int) -> bool:
    return (state == 3 or state == 7 or state == 11 or state == 15)
# ===== END BORDER DETECT FUNCTIONS =====

def getNextState(state: int, action: Action) -> int:
    # move UP
    if (action == Action.up):
        # attempting to move past the top border doesn't change the state
        if (isTopBorder(state)):
            return state
        else:
            return state - 4
    elif (action == Action.right):
        # attempting to move past the right border doesn't change the state
        if (isRightBorder(state)):
            return state
        else:
            return state + 1
    elif (action == Action.down):
        # attempting to move past the bottom border doesn't change the state
        if (isBottomBorder(state)):
            return state
        else:
            return state + 4
    elif (action == Action.left):
        # magic teleporter to terminal state
        if (state == 8):
            return 15
        # attempting to move past the left border doesn't change the state
        elif (isLeftBorder(state)):
            return state
        else:
            return state - 1
    # this shouldn't happen
    else:
        raise RuntimeError("invalid action passed into nextState()")

def calculatePolicyValue(state: int, values: np.ndarray, policy: np.ndarray) -> float:
    value = 0

    if (state == 15):
        return 0

    action = policy[state]
    adjacentState = getNextState(state, action)
    value += reward(state, action)
    value += gamma * values[adjacentState]

    return value

# gradient-ascent-type-thing that takes into account the value of the potential next state + reward for moving there
def getBestAction(state: int, values: np.ndarray, policy: np.ndarray) -> Action:
    maxValue = -999999
    maxValueAction = None

    for action in Action:
         nextState = getNextState(state, action)
         stateValue = calculatePolicyValue(nextState, values, policy)
         if (stateValue > maxValue):
             maxValue = stateValue
             maxValueAction = action

    return maxValueAction

def getBestStochasticAction(state: int, values: np.ndarray) -> StocAction:
    maxValue = -999999
    maxValueActions = []

    for action in Action:
        nextState = getNextState(state, action)
        stateValue = values[nextState]
        # track the action with the highest value
        if (stateValue > maxValue):
            maxValue = stateValue
            maxValueActions = [action]
        # if we find any actions with the same (highest) value, track all of them
        elif (abs(stateValue - maxValue) < 0.000000000001):
            maxValueActions.append(action)

    return StocAction(maxValueActions)

def calcualateNumberOfDeterministicPolicies(stocPolicy: list[StocAction]) -> int:
    numPolicies = 1
    for stocAction in stocPolicy:
        numPolicies *= len(stocAction.actions)

    return numPolicies

def printValues(values: np.ndarray):
    print("values:")
    print(np.reshape(values, (4,4)))
    print("")

def printPolicy(policy: np.ndarray, name: str = ""):
    if (name == ""):
        print("policy: ")
    else:
        print(name + " policy: ")
    print(np.reshape(policy, (4,4)))
    print("")

def printValuesPolicies(values: np.ndarray, policy: np.ndarray, stocPolicy: np.ndarray):
    printValues(values)
    printPolicy(policy)
    printPolicy(stocPolicy, "stochastic")

def main():
    # 1. Initialization
    policy = np.empty(numStates, dtype = Action)
    values = np.zeros(numStates)
    policyStable = False
    iterations = 0

    for i in range(len(policy)):
        policy[i] = random.choice(list(Action))

    while (not policyStable):
        # 2. Policy Evaluation
        delta = 99999
        values2 = np.zeros(numStates)
        while (delta >= theta):
            delta = 0
            for state in range(numStates):
                previousValue = values[state]
                values2[state] = calculatePolicyValue(state, values, policy)
                delta = max(delta, abs(previousValue - values2[state]))

            values = deepcopy(values2)
                
        # 3. Policy Improvement
        policyStable = True
        for state in range(numStates):
            oldAction = policy[state]
            policy[state] = getBestAction(state, values, policy)
            if (oldAction != policy[state]):
                policyStable = False
    
        iterations += 1

    stocPolicy = np.empty(numStates, dtype = StocAction)
    for state in range(numStates):
        stocPolicy[state] = getBestStochasticAction(state, values)

    printValuesPolicies(values, policy, stocPolicy)
    print("Stochastic policy uses the same values as the first deterministic policy")
    print("")
    print("Calculated from the stochastic policy, there are " + str(calcualateNumberOfDeterministicPolicies(stocPolicy)) + " deterministic policies")
    print("")
    print("finished after " + str(iterations) + " iterations")

if (__name__ == "__main__"):
    main()
