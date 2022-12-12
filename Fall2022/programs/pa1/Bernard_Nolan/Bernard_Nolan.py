#Nolan Bernard
import numpy as np
from enum import Enum

class action(Enum):
    left = 0 #left
    up = 1 #up
    right = 2 #right
    down = 3 #down

rewards = np.array([[-1,-1,-1,-1],[-1,-1,-1,-1],[-1,-1,-1,-1],[-1,-1,-1,-1],
                    [-1,-1,-1,-1],[-1,-1,-1,-1],[-1,-1,-1,-1],[-1,-1,-1,-1],
                    [-2,-1,-1,-1],[-1,-1,-1,-1],[-1,-1,-1,-1],[-1,-1,-1,-1],
                    [-1,-1,-1,-1],[-1,-1,-1,-1],[-1,-1,-1,-1],[-1,-1,0,0],])
states = np.array([[0,0,1,4],[0,1,2,5],[1,2,3,6],[2,3,3,7],
                   [4,0,5,8],[4,1,6,9],[5,2,7,10],[6,3,7,11],
                   [15,4,9,12],[8,5,10,13],[9,6,11,14],[10,7,11,15],
                   [12,8,13,12],[12,9,14,13],[13,10,15,14],[14,11,15,15]])    

#initalize policy to be random
policy = np.ones((16,4))*.25
#initialize values to zeros
values = np.zeros(16)

delta = 50 #set the delta to some arbitrary value
epsillon = .01
#loop until it converges within the epsillon value
while(delta > epsillon):
    #update values
    values_old = np.copy(values)
    for i in range(16):
        values[i] = np.dot(policy[i],(rewards[i]+0.95*values[states[i]]))
    delta = np.max(abs(values_old - values))
    #update policy
    for i in range(16):
        policy[i] = np.copy(np.zeros(4))
        policy[i][np.argmax(values[states[i]])] = 1
        
#normalize value function with state 15 to zero
print("Value Function:")
values = values - values[15]
#print the value function
for i in range(4):
    for j in range(4):
        print("{:5.2f}".format(values[4*i+j]), end= ' ')
    print('\n')
    
print("Optimal Deterministic  & Stochastic Policy:")
#print the deterministic policy
for i in range(16):
    print(action(np.argmax(policy[i])).name, end = ' ')
    if (i%4 == 3):
        print('\n')
              
print("Number of Optmial Deterministic Policies:")
total = 1
for i in range(16):
    #find the number of optimal actions in each state
    max_value = max(values[states[i]])
    #total the number of optimal actions for this particular state
    indices = [index for index, value in enumerate(values[states[i]]) if value == max_value]
    #multiply them together to find the total number of optimal determinstic policies
    total = total * len(indices)
print(total)

#The optimal deterministic policy is also ONE optimal stochastic policy