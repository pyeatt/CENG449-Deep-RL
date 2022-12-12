
def move(n, dir):
    if(dir == 0): # move up
        if(n>3):
            n-=4
        return(n)
    
    if(dir == 1):# move right
        if(n%4 != 3):
            n+=1
        return(n)

    if(dir == 2): # move down
        if(n<12):
             n+=4
        return(n)



    if(dir == 3): #move left
        if(n == 8):
            n = 15
        else:
           if(n%4 != 0):
            n-=1
        return(n)
    
    print('Something went wrong')
    return(0)

def reward(n, dir):
    x = -1
    if(n == 15):
        x = 0
    else: 
        if(n == 8):
            if(dir == 3):
                x = -2
    return(x)
            
    

import numpy as np
list1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # to create vector of 0 doubles
list2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # to create vectors of 0 ints

values = np.array(list1)

while True: # Run through once seperately while all probabilites are .25 before making policy deterministic
    new_values = np.array(list1)
    for c  in range(16):
        for action in range(4):
            new_values[c] += (.25 * (.9 * values[move(c, action)] + reward(c, action))) # Bellman
   
    if(np.sum(np.abs(values - new_values)) < .001):
        break # break once convergence is reached
    values = new_values


policy = np.array(list2)
while True:
    new_policy = np.array(list2)
    for c in range(16): #Loop through all 16 states, greedify policy
        max = -9999999.9
        for action in range(4):
            if(values[move(c, action)] + reward(c, action) > max):
                new_policy[c] = action
                max = values[move(c, action)] + reward(c, action)

    e = 0
    for c in range(16):
        if(policy[c] != new_policy[c]):
            e = 1
    if(e ==0):
        break #break if no change from last policy
    policy = new_policy

    while True: #Loop through with deterministic policy
        new_values = np.array(list1)
        for c  in range(16): 
            new_values[c] += (.9 * values[move(c, policy[c])] + reward(c, policy[c])) # Bellman
            
        if(np.sum(np.abs(values - new_values)) < .001):
            break # break once convergence is reached
        values = new_values


print('(One) optimal policy: ')
print(policy)
print('Value function for optimal policy:')
while True: #Loop through with deterministic policy
        new_values = np.array(list1)
        for c  in range(16): 
            new_values[c] += (.9 * values[move(c, policy[c])] + reward(c, policy[c])) # Bellman
            
        if(np.sum(np.abs(values - new_values)) < .001):
            print(values)
            break # break once convergence is reached
        values = new_values

