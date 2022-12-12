"""
    CSC 549 Programming Assignment 1
    Author: Caleb Ehrisman

    0   1   2   3
    4   5   6   7
    8   9   10  11
    12  13  14  15

    The dynamics of this variation of the Gridworld are described as follows:
• States 8 and 15 are special.
    – State 8 is a “magic teleporter”, but using it is expensive. In state 8, the LEFT action takes you
        to state 15 with probability 1 and immediate reward -2.
    – In state 15, the RIGHT and DOWN actions take you back to state 15 with probability 1 and
        immediate reward 0.
• In all states other than states 8 and 15, choosing an action that would move you off the grid will move
you back into the original state with a reward of -1.
• In all remaining cases, choosing an action will move you to an adjacent state within the grid with
probability 1 and an immediate reward of -1. For example, choosing the DOWN action in state 5 will
take you to state 9 with probability 1 and immediate reward -1.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

SIZE = 4
TELE_STATE = [2, 0]
FINAL_STATE = [SIZE - 1, SIZE - 1]
GAMMA_DISCOUNT = 0.95
ACTION_PROB = 0.25
NUM_ITERATIONS = 6

actions = ['L', 'U', 'R', 'D']
ACTIONS_FIGS = ['←', '↑', '→', '↓']

states = [[i, j] for i in range(SIZE) for j in range(SIZE)]
valueMap = np.zeros((SIZE, SIZE))

'''
Author: Caleb Ehrisman
Rewards and edge case handling for future states. 
'''


def reward_policy():
    nextState = []
    actionReward = []
    for i in range(0, SIZE):
        nextState.append([])
        actionReward.append([])
        for j in range(0, SIZE):
            next = dict()
            reward = dict()

            if i == 0:
                next['U'] = [i, j]
                reward['U'] = -1.0
            else:
                next['U'] = [i - 1, j]
                reward['U'] = -1.0
            if i == SIZE - 1:
                next['D'] = [i, j]
                reward['D'] = -1.0
            else:
                next['D'] = [i + 1, j]
                reward['D'] = -1.0
            if j == 0:
                next['L'] = [i, j]
                reward['L'] = -1.0
            else:
                next['L'] = [i, j - 1]
                reward['L'] = -1.0
            if j == SIZE - 1:
                next['R'] = [i, j]
                reward['R'] = -1.0
            else:
                next['R'] = [i, j + 1]
                reward['R'] = -1.0
            if [i, j] == TELE_STATE:
                next['L'] = FINAL_STATE
                reward['L'] = -2.0
            if [i, j] == FINAL_STATE:
                next['R'] = next['D'] = FINAL_STATE
                reward['R'] = reward['D'] = 0.0
            nextState[i].append(next)
            actionReward[i].append(reward)
    return nextState, actionReward


'''
Author: Caleb Ehrisman
Solving the bellman equation by converging on a solution. You can also determine how many 
deterministic solutions there are by this functions output. This does allow for multiple
possible moves from each state that result in the same overall reward score. 

'''


def policy_stochastic(gamma, valueMap):
    env = valueMap

    next_state, reward = reward_policy()
    while True:
        newEnv = np.zeros((SIZE, SIZE))
        for i in range(0, SIZE):
            for j in range(0, SIZE):
                Vs = []
                for action in actions:
                    finalPosition = next_state[i][j][action]
                    Vs.append(reward[i][j][action] + gamma * env[finalPosition[0], finalPosition[1]])
                newEnv[i][j] = np.max(Vs)

        if np.sum(np.abs(env - newEnv)) < 0.0004:
            print("Optimal Stochastic Policy")
            print(newEnv)

            newEnv.reshape(SIZE, SIZE)
            draw_policy(newEnv.reshape(SIZE, SIZE))

            plt.savefig('stochasticPolicy.png')
            plt.close()
            break
        env = newEnv


'''
Author: Caleb Ehrisman

For the this policy Dr. Pyeatt mentioned using a linear system of equations and solving for the unknowns. 
So that is what is being done here. This results in a policy map that there is one best move per state.
'''


def deterministic_policy():
    identity = np.zeros((SIZE * SIZE, SIZE * SIZE))
    np.fill_diagonal(identity, 1)
    M = -1 * identity
    b = np.zeros(SIZE * SIZE)

    next_state, reward = reward_policy()
    for i in range(SIZE):
        for j in range(SIZE):
            curr_state = [i, j]
            index_state = np.ravel_multi_index(curr_state, (SIZE, SIZE))
            for action in actions:
                actionState = next_state[i][j][action]
                actionReward = reward[i][j][action]
                indexStateNew = np.ravel_multi_index(actionState, (SIZE, SIZE))
                M[index_state, indexStateNew] += 0.25 * GAMMA_DISCOUNT
                b[index_state] -= 0.25 * actionReward
    M_inv = np.linalg.inv(M)
    x = np.dot(M_inv, b)

    print("Optimal Deterministic Policy")
    x = np.round(x.reshape(SIZE, SIZE), decimals=3)
    print(x)

    x.reshape(SIZE, SIZE)
    draw_policy(x.reshape(SIZE, SIZE))
    plt.savefig('determinPolicy.png')
    plt.close()


'''
Author: Caleb Ehrisman

https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter03/grid_world.py 

Above link from github repo found in slides. Base Code to draw the policy map nicely. 

'''


def draw_policy(policy):
    fig, axis = plt.subplots()
    axis.set_axis_off()
    tab = Table(axis, bbox=[0, 0, 1, 1])

    row, cols = policy.shape
    width, height = 1 / cols, 1 / row
    next, reward = reward_policy()

    for (i, j), val in np.ndenumerate(policy):
        next_val = []
        for action in actions:
            next_state = next[i][j][action]
            _ = reward[i][j]
            next_val.append(policy[next_state[0], next_state[1]])

        best_ = np.where(next_val == np.max(next_val))[0]
        val = ''
        for ba in best_:
            val += ACTIONS_FIGS[ba]

        tab.add_cell(i, j, width, height, text=val, loc='center', facecolor='white')

    for i in range(len(policy)):
        tab.add_cell(i, -1, width, height, text=i + 1, loc='right', edgecolor='none', facecolor='none')

        tab.add_cell(-1, i, width, height / 2, text=i + 1, loc='center', edgecolor='none', facecolor='none')

    axis.add_table(tab)

'''
Author: Caleb Ehrisman

Shows the Policy Maps using pyplot.imshow

'''
def show_pictures():
    # reading images
    Image1 = plt.imread('determinPolicy.png')
    Image2 = plt.imread('stochasticPolicy.png')

    # showing image
    plt.imshow(Image1)
    plt.axis('off')
    plt.title("Deterministic Policy Map")
    plt.show()

    # showing image
    plt.imshow(Image2)
    plt.axis('off')
    plt.title("Stochastic Policy Map")

    plt.show()


if __name__ == '__main__':
    policy_stochastic(GAMMA_DISCOUNT, valueMap)
    deterministic_policy()
    print("Going by the stochastic policy map you can see there is multiple squares" +
          " that have more than one optional action.\nBy doing 1x2x2x1..1x1x1x2 we can " +
          "determine the number of deterministic polices possible. \nFor this grid enviroment" +
          " there are 128 polices.")
    show_pictures()
