# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 18:01:14 2022

@author: 7450339
"""

import numpy as np
import matplotlib.pyplot as plt

GAMMA = 1.0
LAMBDA = 0.9
ALPHA = 0.001
EPSILON = 0.0
num_episodes = 40
num_trials = 1
fourier_bases_order = 3


ci_sub, ci = [], []
alpha_i = []
for c1 in range(0, fourier_bases_order+1):
    for c2 in range(0, fourier_bases_order+1):
        ci.append([c1, c2])
        norm = np.sqrt(c1^2 + c2^2)
        if(norm == 0.):
            alpha_i.append(ALPHA)
        else:
            alpha_i.append(ALPHA/norm)
ci = np.array(ci)
alpha_i = np.array(alpha_i)

w_forward = np.transpose(np.zeros((fourier_bases_order+1)*(fourier_bases_order+1)))
w_backward = np.transpose(np.zeros((fourier_bases_order+1)*(fourier_bases_order+1)))
w_noward = np.transpose(np.zeros((fourier_bases_order+1)*(fourier_bases_order+1)))



actions = [-1, 0, 1] # full throttle backward, no throttle, full throttle forward

def update_state_space(xt, xdott, at):
    end_of_episode = False
    reward  = -1
    xdot_t1 = xdott + 0.001*at + (-0.0025)*np.cos(3*xt)
    if(xdot_t1 < -0.07):
        xdot_t1 = -0.07
    elif(xdot_t1 > 0.07):
        xdot_t1 = 0.07
        
    xt1 = xt + xdot_t1
    
    if(xt1 <= -1.2):
        xdot_t1 = 0
        xt1 = -1.2
        reward = -1
    elif(xt1 >= 0.5):
        end_of_episode = True
        reward = 0
    
    return end_of_episode, xt1, xdot_t1, reward


def get_fourier_bases_phi_vector(xt, xtdot):
    scale_xt = (xt + 1.2)/1.7
    scale_xtdot = (xtdot + 0.07)/0.14
    ang_vector = np.multiply(np.pi, ci.dot(np.array([scale_xt,scale_xtdot])))
    phi_vector = np.cos(ang_vector)
    return phi_vector
        

def get_new_state():
    xt = np.random.uniform(-0.45, -0.55)
    xtdot = np.random.uniform(-0.01, 0.01)
    return xt, xtdot


def get_greedy_action(xt, xtdot):
    phi_vec = get_fourier_bases_phi_vector(xt, xtdot)
    
    forward = np.transpose(w_forward).dot(phi_vec)
    backward = np.transpose(w_backward).dot(phi_vec)
    noward = np.transpose(w_noward).dot(phi_vec)
    
    if(forward >= backward and forward >= noward):
        act = 1
    elif (backward >= forward and backward >= noward):
        act = -1
    elif (noward >= forward and noward >= backward):
        act = 0
    else:
        print("This shouldn't happen")
    return act


def visualise_value_function():
    position = np.linspace(start=-1.2, stop=0.6, num=50)
    velocity = np.linspace(start=-0.07, stop = 0.07, num=50)
    all_values = []
    
    for pos in position:
        values = []
        for vel in velocity: 
            phi_vec = get_fourier_bases_phi_vector(pos, vel)
            value_forward = np.transpose(w_forward).dot(phi_vec)
            value_backward = np.transpose(w_backward).dot(phi_vec)
            value_noward = np.transpose(w_noward).dot(phi_vec)
            if(value_forward >= value_backward and value_forward >= value_noward):
                value = value_forward
            elif(value_backward >= value_forward and value_backward >= value_noward):
                value = value_backward
            elif(value_noward >= value_forward and value_noward >= value_backward):
                value = value_noward
            else:
                value = 0
                print("This shouldn't happen")
            values.append(value)
        all_values.append(values)
        
    
    all_values = np.array(all_values)
    negative_all_values = np.multiply(-1, all_values)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title("Value Function, Order-{}".format(fourier_bases_order), fontsize=20)
    # ax.contour3D(position, velocity, negative_all_values, 100, cmap='binary')
    x, y = np.meshgrid(position, velocity)
    ax.plot_surface(x, y, negative_all_values)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value');
    plt.show()
    
def plot_agent_locus(path_xt, path_xtdot):
    plt.figure()
    plt.title("Position Vs Velocity", fontsize=20)
    plt.xlabel("Position", fontsize=15)
    plt.ylabel("Velocity", fontsize=15)
    plt.plot(path_xt, path_xtdot)

def plot_learning_rate(all_path_count):
    all_path_count = np.array(all_path_count)
    error_path = np.std(all_path_count, axis=0)
    error_mean = np.mean(all_path_count, axis=0)
    
    plt.figure()
    # plt.errorbar(x_points, error_mean, error_path)
    plt.boxplot(all_path_count)
    plt.plot(range(1, len(error_mean)+1), error_mean)
    plt.title("Learning Curve Order-{} (120 trials)".format(fourier_bases_order), fontsize=20)
    plt.xlabel("Number of Episodes", fontsize=15)
    plt.ylabel("Number of Steps Per Episode", fontsize=15)
    plt.show()
    
    
def update_weights_and_z(phi_vec, new_phi_vec, w, z, reward, q_old):
    q = np.transpose(w).dot(phi_vec)
    new_q = np.transpose(w).dot(new_phi_vec)
        
    delta = reward + GAMMA*new_q - q

    new_z = GAMMA*LAMBDA*z+(1-alpha_i*GAMMA*LAMBDA*(np.transpose(z)@phi_vec))*phi_vec
    new_w = w+ alpha_i*(delta+q-q_old)*new_z- alpha_i*(q-q_old)*phi_vec
    
    return new_z, new_w, new_q
    

all_path_count = []
for j in range(0, num_trials):
    print(j)
    path_xt, path_xtdot = [], []
    path_count = []
    w_forward = np.transpose(np.zeros((fourier_bases_order+1)*(fourier_bases_order+1)))
    w_backward = np.transpose(np.zeros((fourier_bases_order+1)*(fourier_bases_order+1)))
    w_noward = np.transpose(np.zeros((fourier_bases_order+1)*(fourier_bases_order+1)))
    for i in range(0, num_episodes):
        end_of_episode = False
        xt, xtdot = get_new_state()
        path_xt.append(xt)
        path_xtdot.append(xtdot)
    
        ac = get_greedy_action(xt, xtdot)
        phi_vec = get_fourier_bases_phi_vector(xt, xtdot) 
        z_forward = np.transpose(np.zeros((fourier_bases_order+1)*(fourier_bases_order+1)))
        z_backward = np.transpose(np.zeros((fourier_bases_order+1)*(fourier_bases_order+1)))
        z_noward = np.transpose(np.zeros((fourier_bases_order+1)*(fourier_bases_order+1)))
        q_old = 0
        count = 0
        while(end_of_episode == False):
            end_of_episode, new_xt, new_xtdot, reward = update_state_space(xt, xtdot, ac)
            path_xt.append(new_xt)
            path_xtdot.append(new_xtdot)
                
            if(end_of_episode == True):
                path_count.append(count)
                break
            
            new_ac = get_greedy_action(new_xt, new_xtdot)
            new_phi_vec = get_fourier_bases_phi_vector(new_xt, new_xtdot)
            
            if( ac == -1):
                z_backward, w_backward, new_q = update_weights_and_z(phi_vec, new_phi_vec, w_backward, z_backward, reward, q_old)
                z_forward = GAMMA*LAMBDA*z_forward
                z_noward = GAMMA*LAMBDA*z_noward
                
            elif(ac == 0):
                z_noward, w_noward, new_q = update_weights_and_z(phi_vec, new_phi_vec, w_noward, z_noward, reward, q_old)
        
                z_forward = GAMMA*LAMBDA*z_forward
                z_backward = GAMMA*LAMBDA*z_backward
                
            elif(ac == 1):
                z_forward, w_forward, new_q = update_weights_and_z(phi_vec, new_phi_vec, w_forward, z_forward, reward, q_old)
        
                z_noward = GAMMA*LAMBDA*z_noward
                z_backward = GAMMA*LAMBDA*z_backward
            
            else:
                print("This shouldn't happen")
            q_old = new_q
            phi_vec = new_phi_vec
            ac = new_ac
            
            xt = new_xt
            xtdot = new_xtdot
           
            count = count + 1
    all_path_count.append(path_count)
    
visualise_value_function()
plot_learning_rate(all_path_count)