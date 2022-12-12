"""
Author: Alex Hanson
Date: 11/26/2022
"""
import numpy as np
from time import time
import argparse
from matplotlib import pyplot as plt # graphing
from matplotlib import cm
from render import render # function to render postition list to animation
from color_text import * # color printing

# ==========================================================================================


from mountain_car_simulator import MountainCarSimulator
import random

ACTIONS = [-1, 0, 1]    # left, nothing, right
num_actions = 3
state_dimensions = 2
max_timesteps = 1000



def generate_fourier(order: int): # only works for a state space with dimensions of 2
    values = np.array([x for x in range(order + 1)])
    col_1 = np.repeat(values, order + 1)
    col_2 = np.tile(values, order + 1)
    return np.column_stack((col_1, col_2))

def features(F, state): # compute the Fourier Basis features
    return np.cos(np.pi * np.dot(F, state))

def normalize(state):
    """
    Converts state to between 0-1
    Returns new normalized state
    """
    lower_bounds = np.array([-1.2, -0.07])
    upper_bounds = np.array([0.6, 0.07])
    s_ = (state - lower_bounds)/(upper_bounds - lower_bounds)
    return s_


def epsilon_greedy_action(F, weights, state, epsilon):
    if random.random() <= epsilon:
        return random.randrange(0, num_actions)

    q_values = []
    for action in range(num_actions):
        q_values.append(np.dot(weights[:, action], features(F, state)))
    index = np.argmax(q_values)
    return index

def V(F, weights, state):
    q_values = []
    for a in range(num_actions):
        q_values.append(np.dot(weights[:, a], features(F, state)))
    return np.sum(q_values)



def run(num_episodes,   # Episode count
        order_fourier,  # Order x fourier base
        gamma = 1.0,    # discount factor
        alpha = 0.001,  # learning rate
        epsilon = 0.01, # exploration rate
        lambda_ = 0.9): # decay rate

    env = MountainCarSimulator() # Set up environment


    num_basis_functions = (order_fourier + 1)**2

    weights = np.zeros((num_basis_functions, num_actions))
    e = np.zeros((num_basis_functions, num_actions))
    F = generate_fourier(order_fourier)

    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
    with np.errstate(divide='ignore'): # ignore divide by zero error when constructing
        # gradient descent a_i = a_1/||c_i||_2
        basis_functions_learning_rate = alpha / np.linalg.norm(F, ord=2, axis=1) 
    basis_functions_learning_rate[0] = alpha # avoiding division by zero a_0 = a_1

    p_list = []
    return_values = []

    for episode in range(num_episodes):
        p_list = []
        accumulated_reward = 0
        step = 0

        e[:] = 0                                # Clear traces
        s = normalize(env.initilize())          # reset environment
        a = epsilon_greedy_action(F, weights, s, epsilon)# pick action from state
        for timestep in range(max_timesteps):
            step += 1
            # print(f"{episode},{timestep} {accumulated_reward} taking action {a} in state {env.getState()} -> {s}")
            # print(f"{env.getState()[0]},", end='')
            p_list.append(env.getState()[0]) # used for animation
            
            e[:] *= gamma * lambda_             # decay traces
            s_prime, r = env.step(ACTIONS[a])   # actually take action a, observe s` and r
            s_prime = normalize(s_prime)
            terminal = env.isTerminal()
            a_prime = epsilon_greedy_action(F, weights, s_prime, epsilon) # pick action from s`

            accumulated_reward += r

            # SARSA============================================
            # compute basis function for state and action pairs
            basis_functions_s = features(F, s)
            basis_functions_sprime = features(F, s_prime)

            e[:, a] += basis_functions_s  # accumulate traces

            delta = r - np.dot(weights[:, a], basis_functions_s) # TD error, delta = reward - Q(s,a)
            if terminal:
                for action in range(num_actions):
                    # update weights
                    weights[:, action] += delta * np.multiply(basis_functions_learning_rate, e[:, action])
                break

            q_value = np.dot(weights[:, a_prime], basis_functions_sprime) 
            delta = delta + gamma * q_value #delta = delta + gamma * Q(s`,a`)

            for action in range(num_actions):
                # update weights
                weights[:, action] += delta * np.multiply(basis_functions_learning_rate, e[:, action])
            # end SARSA========================================
            s = s_prime
            a = a_prime
            

        print(f"End episode {episode}, reward: {accumulated_reward}, position: {env.getState()[0]:.5f}")
        return_values.append(step)
        # uncomment this to generate animation at the end of training
        # if episode+1 == num_episodes:
        #     render("out_" + str(order_fourier), p_list)

    return return_values, F, weights













# returns steps averaged over samples
def sample(order_fourier = 3, num_samples = 50, episodes_count = 21):
    s_t = time()
    steps = []
    for sample in range(num_samples):
        start_time = time()
        results, _, _ = run(episodes_count, order_fourier)
        end_time = time()
        duration = end_time - start_time
        printGreen(f"{episodes_count} episodes ran over {duration:.5f} seconds")
        steps.append(results)
    steps = np.array(steps)
    averages = np.average(steps, axis=0)
    e_t = time()
    print(f"{num_samples} samples ran over {e_t - s_t:.5f}")
    print("Average steps for episode:")
    printCyan(averages)
    return averages







def part_1():
    episodes_count = 21
    num_samples = 100
    # fixed alpha, epsilon, gamma = 1, lambda = 0.9
    # Generate learning curve order 3
    steps_3 = sample(3, num_samples, episodes_count)
    # Generate learning curve order 5
    steps_5 = sample(5, num_samples, episodes_count)
    # Generate learning curve order 7
    steps_7 = sample(7, num_samples, episodes_count)

    # plot order 3
    plt.figure(1)
    plt.plot(np.arange(0,episodes_count), steps_3, color = 'red')
    plt.title("O(3) Fourier")
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.xlim(0, episodes_count - 1)
    plt.xticks(np.arange(0,episodes_count,(episodes_count-1)/10))
    # plot order 5
    plt.figure(2)
    plt.plot(np.arange(0,episodes_count), steps_5, color = 'red')
    plt.title("O(5) Fourier")
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.xlim(0, episodes_count - 1)
    plt.xticks(np.arange(0,episodes_count,(episodes_count-1)/10))
    # plot order 7
    plt.figure(3)
    plt.plot(np.arange(0,episodes_count), steps_7, color = 'red')
    plt.title("O(7) Fourier")
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.xlim(0, episodes_count - 1)
    plt.xticks(np.arange(0,episodes_count,(episodes_count-1)/10))

    # plot all 1 graph
    plt.figure(4)
    plt.plot(np.arange(0,episodes_count), steps_3, color = 'red', label = "O(3)")
    plt.plot(np.arange(0,episodes_count), steps_5, color = 'blue', label = "O(5)")
    plt.plot(np.arange(0,episodes_count), steps_7, color = 'green', label = "O(7)")
    #plt.legend(["O(3)", "O(5)", "O(7)"], loc=4)
    plt.legend()
    plt.title("Learning Curve")
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.xlim(0, episodes_count - 1)
    plt.xticks(np.arange(0,episodes_count,(episodes_count-1)/10))

    plt.show()




def part_2():
    episodes_count = 1000
    # over 1000 episodes
    # Generate surface plot value function order 3
    steps, F_3, weights_3 = run(episodes_count, 3)
    # Generate surface plot value function order 5
    steps, F_5, weights_5 = run(episodes_count, 5)
    # Generate surface plot value function order 7
    steps, F_7, weights_7 = run(episodes_count, 7)

    
    precision = 64#32

    # x - position -1.2 to 0.6
    x = np.outer(np.linspace(-1.2, 0.6, precision), np.ones(precision))
    # y - velocity -0.7 to 0.7
    y = np.outer(np.linspace(-0.7, 0.7, precision), np.ones(precision)).T
    # z - value function
    z = np.zeros(x.shape)

    x_2 = np.outer(np.linspace(0.0, 1.0, precision), np.ones(precision))
    y_2 = np.outer(np.linspace(0.0, 1.0, precision), np.ones(precision)).T
    for i in range(precision):
        for j in range(precision):
            st = np.array([x_2[i][j], y_2[i][j]])
            z[i][j] = -1 * V(F_3, weights_3, st)#np.dot(weights_3[:, 2], features(F_3, st)) #V(F, weights, st)
            #z[i][j] = -1 * V(F_3, weights_3, [x[i][j], y[i][j]])

    # plot order 3
    plt.figure(1)
    ax = plt.axes(projection="3d")
    ax.plot_surface(x,y,z, cmap=cm.jet)
    plt.title("O(3) Value Function 1000 episodes")
    plt.xlabel("Position")
    plt.ylabel("Velocity")

    # plot order 5
    plt.figure(2)
    for i in range(precision):
        for j in range(precision):
            st = np.array([x_2[i][j], y_2[i][j]])
            z[i][j] = -1 * V(F_5, weights_5, st)
            # z[i][j] = -1 * V(F_5, weights_5, [x[i][j], y[i][j]])
    ax = plt.axes(projection="3d")
    ax.plot_surface(x,y,z, cmap=cm.jet)
    plt.title("O(5) Value Function 1000 episodes")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    # plot order 7
    plt.figure(3)
    for i in range(precision):
        for j in range(precision):
            st = np.array([x_2[i][j], y_2[i][j]])
            z[i][j] = -1 * V(F_7, weights_7, st)
            # z[i][j] = -1 * V(F_7, weights_7, [x[i][j], y[i][j]])
    ax = plt.axes(projection="3d")
    ax.plot_surface(x,y,z, cmap=cm.jet)
    plt.title("O(7) Value Function 1000 episodes")
    plt.xlabel("Position")
    plt.ylabel("Velocity")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part_1", action='store_true', help="Generate graphs for part 1. Learning Curves 3,5,7")
    parser.add_argument("--part_2", action='store_true', help="Generate surface for part 2. Value function 3,5,7")
    # parser.add_argument("-e", "--episodes", default=21, dest="episodes", help="Number of episodes to run.", type=int)
    # parser.add_argument("-s", "--samples", default=50, dest="samples", help="Number of samples to run.", type=int)
    # parser.add_argument("--alpha", default=0.05, dest="alpha", help="Alpha value. Learning rate.", type=float)
    # parser.add_argument("--gamma", default=1.0, dest="gamma", help="Gamma value. Discount factor.", type=float)
    # parser.add_argument("--epsilon", default=0.5, dest="epsilon", help="Epsilon value. Exploration rate.", type=float)
    # parser.add_argument("--fourier", default=3, dest="order_fourier", help="Order Fourier base.", type=int)
    # parser.add_argument("--surface", action='store_true', help="Generate surface plot for value function.")
    # parser.add_argument("--curve", action='store_true', help="Generate learning curve.")
    # filename for saving graphs
    args = parser.parse_args()

    if args.part_1:
        input("Will generate all 3 graphs before showing. Takes a while to run. Press enter to start or ctrl-C to break.")
        part_1()
    elif args.part_2:
        input("Will generate all 3 graphs before showing. Takes a while to run. Press enter to start or ctrl-C to break.")
        part_2()
    else:
        print("Please use -h for usage.")
        # generate sample
        #data = sample(order_fourier=args.order_fourier, episodes_count=args.episodes, num_samples=args.samples)
        #steps, F_3, weights_3 = run(args.episodes, args.order_fourier)





        # episodes_count = 501
        # num_samples = 50

        # steps_3 = sample(3, num_samples, episodes_count)
        # plt.figure(1)
        # plt.plot(np.arange(0,episodes_count), steps_3, color = 'red')
        # plt.title("O(3) Fourier")
        # plt.xlabel("Episodes")
        # plt.ylabel("Steps")
        # plt.xlim(0, episodes_count - 1)
        # plt.xticks(np.arange(0,episodes_count,(episodes_count-1)/10))
        
        # plt.show()

        # episodes_count = 1001
        # steps_3, F_3, weights_3 = run(episodes_count, 3)
        # plt.plot(np.arange(0,episodes_count), steps_3, color = 'red')
        # plt.title("O(3) Fourier")
        # plt.xlabel("Episodes")
        # plt.ylabel("Steps")
        # plt.xlim(0, episodes_count - 1)
        # plt.xticks(np.arange(0,episodes_count,(episodes_count-1)/10))
        # plt.show()


