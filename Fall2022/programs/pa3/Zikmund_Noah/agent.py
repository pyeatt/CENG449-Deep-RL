# Noah Zikmund
# CENG 449
# Dr. Pyeatt
# 2 December 2022

# SARSA LAMBDA with accumulating traces
# Loop for each run
    # Loop for each episode
    # Initialize S normalized between [0, 1] (For the Fourier Basis)
        # Choose A ~ randomly or epsilon-greedy(S, weights)
        # z <- 0

        # epsilon-greedy
            # If rand <= epsilon
                # Choose random A
            # Else
                # Choose max q(S, A, w) => w(A) dot product feature-vector(S)

        # Loop for each step of episode
            # Z(A) += Phi(S) aka feature vector
            # Take action A, observe R, S'
            # d <- R - q(S, A, w)

            # If S' is terminal then:
                # w <- w + alpha * (delta) * z
                # break
            # Else choose A'~ pi(|S) or epsilon-greedy(S, weights)
            # d += gamma * q(S', A', w)
            # w <- w + alpha * (delta) * z
            # z <- gamma * lambda * z
            # S <- S'
            # A <- A'

import gym
import numpy as np
import random
import math
from MountainCar import MountainCar
from matplotlib import pyplot as plt
from matplotlib import cm

ALPHA = 0.001       # (learning rate)
GAMMA = 1	        # (discount rate)
LAMBDA = 0.9	    # (trace paramater)
EPSILON = 0         # (totally greedy)
MAX_STEPS = 200     # (max episode length)
EPISODES = 1000     # (number of episodes)
ANIMATE = 0         # (boolean to animate or not)
RUNS = 1            # (number of runs)
ORDER = 5           # (the order for the run (3, 5, or 7))

def get_coefficients():
    temp1 = 0
    temp2 = 0
    C = np.zeros(((ORDER + 1)**2, 2))
    for i in range((ORDER + 1)**2):
        C[i][0] = temp2
        C[i][1] = temp1
        temp1 = temp1 + 1
        if(temp1 == ORDER + 1):
            temp1 = 0
            temp2 = temp2 + 1
    return C

def get_weights():
    w = np.zeros(((ORDER + 1)**2, 3))
    return w

def get_action(phi, w, boolean):

    num_actions = 3
    best = float("-inf")
    best_actions = []

    if(boolean):
        actions = [0, 1, 2]
        return random.choice(actions)

    for a in range(num_actions):
        this_Q = get_value(phi, a, w)

        if(math.isclose(this_Q, best)):
            best_actions.append(a)
        elif(this_Q > best):
            best = this_Q
            best_actions = [a]  
    
    action = random.choice(best_actions)
    return action

def get_value(phi, A, w):
    return np.dot(phi, w[:, A])   
        
def get_basis_function(state_vector):
    coefficients = get_coefficients()
    basis_func = np.cos(np.pi*(np.dot(coefficients, state_vector)))
    return basis_func        

def main():
    global ALPHA, GAMMA, LAMBDA, EPSILON, MAX_STEPS, EPISODES, RUNS, ANIMATE, ORDER

    run_data = np.zeros(EPISODES)
    x_axis = np.zeros(EPISODES)
    y_axis = np.zeros(EPISODES)
    z_axis = np.zeros(EPISODES)

    model = MountainCar()                               # Get MountainCar Class
    num_bases = (ORDER + 1)**2                          # Based on ORDER, the number of basis funcs

    for runs in range(RUNS):
        steps = 0
        w = get_weights()                              # Inits the weights to zero and proper size                        
        for x in range(EPISODES):
            model.reset()                                   # Init MountainCar
            if(ANIMATE):
                model.animate(x, 0, MAX_STEPS)

            S = model.getState()                            # Initialize S
            phi = get_basis_function(model.normalize(S))
            A = get_action(phi, w, 0)                       # Get action A based on e-greedy or random
            z = np.zeros((num_bases, 3))                    # init z = 0  

            for y in range(MAX_STEPS):
                steps = steps + 1
                S_norm = model.normalize(S)
                phi = get_basis_function(S_norm)     # get feature-vector of fourier basis funcs
                z[:, A] += phi                              # trace
                S_next, R, terminated = model.step(A)       # returns next state, reward, and if-terminated

                if(ANIMATE):
                    model.animate(x, y + 1, MAX_STEPS)

                q = get_value(phi, A, w)
                d = R - q

                if(terminated):
                    w = w + (ALPHA * d * z)
                    break
                else:
                    S_norm = model.normalize(S_next)
                    phi = get_basis_function(S_norm)
                    A_next = get_action(phi, w, 0)

                    q_next = get_value(phi, A_next, w)
                    d = d + (GAMMA * q_next)
                    w = w + (ALPHA * d * z)
                    z = z * (GAMMA * LAMBDA)
                    S = S_next
                    A = A_next
                    

            run_data[x] += steps
            steps = 0

    #-----------------------------------------------#
    #           Learning Curve Plot
    for x in range(EPISODES):
        run_data[x] = run_data[x] / RUNS


    data_range = range(0, EPISODES)

    plt.plot(data_range, run_data)
    plt.title("MountainCar FB Order " + str(ORDER))
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.show()
    #
    #-----------------------------------------------#

    #-----------------------------------------------#
    #          Value Cost Plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x1samp = np.linspace(-1.2, 0.6, 100)
    x2samp = np.linspace(-0.07, 0.07, 100)

    x_axis, y_axis = np.meshgrid(x1samp, x2samp)
    z_axis = np.zeros(10000)

    for i in range(100):
        for j in range(100):
            state = [x_axis[i, j], y_axis[i, j]]
            phi = get_basis_function(model.normalize(state))
            action = get_action(phi, w, 0)
            
            temp_Q = get_value(phi, action, w)
            z_axis[j + i*100] = -1.0*temp_Q

    ax.scatter(x_axis, y_axis, z_axis)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost Value')
    ax.set_zlim(0, 150)
    ax.set_title(f'O({ORDER})\n\n{EPISODES} Episodes')
    plt.show()
    #
    #-----------------------------------------------#


if __name__ == "__main__":
    main()

