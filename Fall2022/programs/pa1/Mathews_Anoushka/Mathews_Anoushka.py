# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 13:17:09 2022

@author: Anoushka Mathews
"""

# I am using Python Version 3.8

# When you run this script you should see 2 plots. 
# A plot for the deterministic policy with its value function
# and a plot for stochastic policy with its value function. 
# Both value functions are the same. 
# Also, you should see a print statement on the console with 
# the number of deterministic policies in this system for a particular gamma. 
# You can change the gamma variable to some other value if you'd like. 

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt


STATE_VALUES_V = np.zeros(16)
gamma = 0.95

class STATE(Enum):
    ST0 = 0
    ST1 = 1
    ST2 = 2
    ST3 = 3
    ST4 = 4
    ST5 = 5
    ST6 = 6
    ST7 = 7
    ST8 = 8
    ST9 = 9
    ST10 = 10
    ST11 = 11
    ST12 = 12
    ST13 = 13
    ST14 = 14
    ST15 = 15
    
    
class ACTION(Enum):
    TOP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    
def new_state(state, action):
    new_state = state
    if(action == ACTION.TOP):
        if(state == STATE.ST4):
            new_state = STATE.ST0
        elif (state == STATE.ST5):
            new_state = STATE.ST1
        elif (state == STATE.ST6):
            new_state = STATE.ST2
        elif (state == STATE.ST7):
            new_state = STATE.ST3
            
        elif(state == STATE.ST8):
            new_state = STATE.ST4
        elif (state == STATE.ST9):
            new_state = STATE.ST5
        elif (state == STATE.ST10):
            new_state = STATE.ST6
        elif (state == STATE.ST11):
            new_state = STATE.ST7
            
        elif(state == STATE.ST12):
            new_state = STATE.ST8
        elif (state == STATE.ST13):
            new_state = STATE.ST9
        elif (state == STATE.ST14):
            new_state = STATE.ST10
        elif (state == STATE.ST15):
            new_state = STATE.ST11
    
    elif(action == ACTION.RIGHT):
        if(state == STATE.ST0):
            new_state = STATE.ST1
        elif (state == STATE.ST4):
            new_state = STATE.ST5
        elif (state == STATE.ST8):
            new_state = STATE.ST9
        elif (state == STATE.ST12):
            new_state = STATE.ST13
            
        elif(state == STATE.ST1):
            new_state = STATE.ST2
        elif (state == STATE.ST5):
            new_state = STATE.ST6
        elif (state == STATE.ST9):
            new_state = STATE.ST10
        elif (state == STATE.ST13):
            new_state = STATE.ST14
            
        elif(state == STATE.ST2):
            new_state = STATE.ST3
        elif (state == STATE.ST6):
            new_state = STATE.ST7
        elif (state == STATE.ST10):
            new_state = STATE.ST11
        elif (state == STATE.ST14):
            new_state = STATE.ST15
            
    elif(action == ACTION.DOWN):
        if(state == STATE.ST0):
            new_state = STATE.ST4
        elif (state == STATE.ST1):
            new_state = STATE.ST5
        elif (state == STATE.ST2):
            new_state = STATE.ST6
        elif (state == STATE.ST3):
            new_state = STATE.ST7
            
        elif(state == STATE.ST4):
            new_state = STATE.ST8
        elif (state == STATE.ST5):
            new_state = STATE.ST9
        elif (state == STATE.ST6):
            new_state = STATE.ST10
        elif (state == STATE.ST7):
            new_state = STATE.ST11
            
        elif(state == STATE.ST8):
            new_state = STATE.ST12
        elif (state == STATE.ST9):
            new_state = STATE.ST13
        elif (state == STATE.ST10):
            new_state = STATE.ST14
        elif (state == STATE.ST11):
            new_state = STATE.ST15
        
    elif(action == ACTION.LEFT):
        if(state == STATE.ST3):
            new_state = STATE.ST2
        elif (state == STATE.ST7):
            new_state = STATE.ST6
        elif (state == STATE.ST11):
            new_state = STATE.ST10
        elif (state == STATE.ST15):
            new_state = STATE.ST14
            
        elif(state == STATE.ST2):
            new_state = STATE.ST1
        elif (state == STATE.ST6):
            new_state = STATE.ST5
        elif (state == STATE.ST10):
            new_state = STATE.ST9
        elif (state == STATE.ST14):
            new_state = STATE.ST13
            
        elif(state == STATE.ST1):
            new_state = STATE.ST0
        elif (state == STATE.ST5):
            new_state = STATE.ST4
        elif (state == STATE.ST9):
            new_state = STATE.ST8
        elif (state == STATE.ST13):
            new_state = STATE.ST12
            
        elif(state == STATE.ST8):
            new_state = STATE.ST15
        
    return new_state

def prob_new_state_given_state_action_det(state, action, state_prime):
    n_state = new_state(state, action)
    if(n_state == state_prime):
        return 1
    else:
        return 0
    

def reward(state, action):
    if(state == STATE.ST8 and action == ACTION.LEFT):
        reward = -2
    elif (state == STATE.ST15 and (action == ACTION.DOWN or action == ACTION.RIGHT)):
        reward = 0
    else:
        reward = -1
    
    return reward

def diff_state_values_v(array1, array2):
    max_diff = 0
    for arr1, arr2 in zip(array1, array2):
        diff = abs(arr1 - arr2)
        if(max_diff < diff):
            max_diff = diff
    return max_diff


def value_iteration_v(theta=0.1):
    delta = np.infty
    max_val = -np.infty
    
    while(delta > theta):
        old_values = STATE_VALUES_V.copy()
        
        for state in STATE:
            max_val = -np.infty

            for action in ACTION:
                val = 0
                r = reward(state, action)
                
                for n_state in STATE:
                    p = prob_new_state_given_state_action_det(state, action, n_state)
                    val = val+p*(r+gamma*old_values[n_state.value])
  
                if(max_val < val):
                    max_val = val
  
            STATE_VALUES_V[state.value] = max_val
        
        delta = diff_state_values_v(old_values, STATE_VALUES_V)
    
    return True


def get_policy_v_det():
    optimal_policy = np.empty(16, dtype=ACTION)
    
    for state in STATE:
        val_of_states = []
        for i, action in enumerate(ACTION):
            n_state = new_state(state, action)
            val_of_states.append(STATE_VALUES_V[n_state.value])
        
        index_of_action = np.array(val_of_states).argmax()
        optimal_policy[state.value] = ACTION(index_of_action)
        
    return optimal_policy


def get_policy_v_sch():
    optimal_policy = np.empty(16, dtype=ACTION)
    state_action_possibilities = []
    
    for state in STATE:
        state_action_possibilities = []
        val_of_states = []
        for i, action in enumerate(ACTION):
            n_state = new_state(state, action)
            val_of_states.append(STATE_VALUES_V[n_state.value])
            

        max_value = np.array(val_of_states).max()
        for i, val in enumerate(val_of_states):
            if(val == max_value):
                state_action_possibilities.append(ACTION(i))
        
        optimal_policy[state.value] = state_action_possibilities
    
    return optimal_policy

def get_number_of_det_policies():
    num_det_pol = 1
    for pol in opt_policy_sch:
        num_det_pol = num_det_pol*len(pol)
    return num_det_pol

def return_enum_names_det(array):
    opt_policy_names = []
    for p_det in array:
        opt_policy_names.append(p_det.name)
    return opt_policy_names

def return_enum_names_sch(array):
    opt_policy_names = []
    for p_det in opt_policy_sch:
        state_action = []
        for i in range(len(p_det)):
            state_action.append(p_det[i].name)
        opt_policy_names.append(state_action)
    return opt_policy_names
        

def present_results(opt_policy_det_names, opt_policy_sch_names):
    
    state_values_2d = np.round(STATE_VALUES_V.reshape([4,4]), 3)
    
    # Show Deterministic Policy
    fig, [ax2,ax1] = plt.subplots(1, 2, figsize=(20,20))
    ax1.imshow(state_values_2d)
    ax1.set_title("Optimal Deterministic Policy", fontsize=20)
    for (j,i),label in np.ndenumerate(np.array(opt_policy_det_names).reshape([4,4])):
        ax1.text(i,j,label,ha='center',va='center', fontsize=20)
        
    ax2.imshow(state_values_2d)
    ax2.set_title("Value Function V*", fontsize=20)
    for (j,i),label in np.ndenumerate(state_values_2d):
        ax2.text(i,j,label,ha='center',va='center', fontsize=20)
    plt.show()
    
    
    # Show Stochastic Policy
    fig, [ax4,ax3] = plt.subplots(1, 2,figsize=(20,30))
    ax3.imshow(state_values_2d)
    ax3.set_title("Optimal Stochastic Policy", fontsize=20)
    for (j,i),label in np.ndenumerate(np.array(opt_policy_sch_names, dtype=list).reshape([4,4])):
        ax3.text(i,j,label,ha='center',va='center', fontsize=12)
        
    ax4.imshow(state_values_2d)
    ax4.set_title("Value Function V*", fontsize=20)
    for (j,i),label in np.ndenumerate(state_values_2d):
        ax4.text(i,j,label,ha='center',va='center', fontsize=20)  
    plt.show()
    
    
    # Print the number of deterministic policies there are
    print("There are a total of", get_number_of_det_policies(),"optimal deterministic policies for the gamma of", gamma)

    return True

value_iteration_v()
opt_policy_det = get_policy_v_det()
opt_policy_det_names = return_enum_names_det(opt_policy_det)
opt_policy_sch = get_policy_v_sch()
opt_policy_sch_names = return_enum_names_sch(opt_policy_sch)
num_det_pol = get_number_of_det_policies()
present_results(opt_policy_det_names, opt_policy_sch_names)