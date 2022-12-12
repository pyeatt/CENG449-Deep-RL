# Karissa Schipke
# Python 3.9.7

from termcolor import colored
import numpy as np
import random
import copy

ROW = 4
COL = 4

TELEPORT = (2, 0)
WIN = (3, 3)

GAMMA = 0.95

class State:
    def __init__(self, location, value = 0.0): #location is tuple (x, y)
        self.location = location
        self.value = value

        self.rr = -1
        self.lr = -1
        self.ur = -1
        self.dr = -1

        if location == TELEPORT:
            self.lr = -2
        elif location == WIN:
            self.rr = 0
            self.dr = 0


class Board:
    def __init__(self):
        self.board = np.empty((ROW, COL), dtype=object)
        for i in range(ROW):
            for j in range(COL):
                self.board[i, j] = State((i, j))

    def printBoard(self):
        print('Value:')
        for i in range(ROW):
            s = '| '
            for j in range(COL):
                s += "{:.2f}".format(self.board[i, j].value)
                s += ' | '
            print(s)
        print('\n')

    def getState(self, location):
        return self.board[location[0], location[1]]


class Agent:
    def __init__(self):
        self.board = Board()
        self.states = []
        self.reward = 0
        self.actions = ["u", "d", "l", "r"]
        self.policy = np.empty((ROW, COL), dtype=object)
        for i in range(ROW):
            for j in range(COL):
                self.policy[i, j] = self.actions

    def printBoard(self):
        self.board.printBoard()

    def printPolicy(self):
        print('Policy:')
        for row in self.policy:
            print('| ', end = '')
            for x, p in enumerate(row):
                print(''.join(p).center(4, " "), end=' | ' if x != COL - 1 else ' |\n')
        print('')

    def checkAction(self, state, action):
        if action == "u":
            row = max(0, state.location[0] - 1)
            nextState = self.board.getState((row, state.location[1]))
        elif action == "d":
            row = min(state.location[0] + 1, ROW - 1)
            nextState = self.board.getState((row, state.location[1]))
        elif action == "l":
            col = max(0, state.location[1] - 1)
            nextState = self.board.getState((state.location[0], col))
            if state.location == TELEPORT:
                nextState = self.board.getState(WIN)
        else:
            col = min(state.location[1] + 1, COL - 1)
            nextState = self.board.getState((state.location[0], col))
        return nextState

    def probability(self, action, policy):
        if action in policy:
            return 1.0 / len(policy)
        return 0

    def policyImprovement(self):
        unstable = False
        for row in self.board.board:
            for state in row:
                old = self.policy[state.location[0], state.location[1]]
                maxValue = float('-inf')
                for a in self.actions:
                    nextState = self.checkAction(state, a)
                    nextValue = nextState.value
        
                    if nextValue > maxValue:
                        action = [a]
                        maxValue = nextValue
                    elif nextValue == maxValue:
                        action.append(a)
                self.policy[state.location[0], state.location[1]] = action
                if not unstable and old != action:
                    unstable = True
        return unstable

    def policyEvaluation(self):
        change = 1
        cp = copy.deepcopy(self.board.board)

        while change > 0:
            change = 0
            for row in self.board.board:
                for state in row:
                    value = state.value
                    state.value = 0

                    for a in self.actions:
                        if a == "u":
                            nextReward = state.ur
                        elif a == "d":
                            nextReward = state.dr
                        elif a == "l":
                            nextReward = state.lr
                        else:
                            nextReward = state.rr

                        nextState = self.checkAction(state, a)
                        state.value += self.probability(a, self.policy[state.location[0], state.location[1]]) * (nextReward + GAMMA * cp[nextState.location[0], nextState.location[1]].value)
                    change = max(change, abs(value - state.value))

            
    def optimalPolicy(self):
        # self.printBoard()
        # self.printPolicy()
        changed = True

        while(changed):
            self.policyEvaluation()
            changed = self.policyImprovement()

        self.printBoard()
        self.printPolicy()

    def printNumDeterministic(self):
        num = 1
        for row in self.policy:
            for p in row:
                num = num * len(p)
        print("Number of optimal deterministic policies: " + str(num) + "\n")

    def printDeterministic(self):
        print('Policy:')
        for row in self.policy:
            print('| ', end = '')
            for x, p in enumerate(row):
                print(''.join(p[0]).center(4, " "), end=' | ' if x != COL - 1 else ' |\n')
        print('')

if __name__ == "__main__":
    agent = Agent()

    print(colored('\nStochastic Policy\n', 'magenta'))
    agent.optimalPolicy()

    print(colored('Deterministic Policy\n', 'magenta'))
    agent.printBoard()
    agent.printDeterministic()
    agent.printNumDeterministic()
