import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns

Velocity = np.linspace(-0.07,0.07,101)
Position = np.linspace(-1.2,0.6,101)
StateRewardVec = []
action = np.array([-1, 0, 1])
#fig = plt.figure()

# fig, [ax1, ax2] = plt.subplots(2,1)
# ax1.clear()
# ax1.plot(x,y)
#plt.show(block=False)

def UpdateSimulation(State, Action = 1):
    curPos = State[0] #Position
    curVel = State[1] #Velocity
    global reward
    global StateRewardVec
    act = action[Action] #Map from 0:2 to -1:1
    #update velocity
    curVel += (act * 0.001 + math.cos(3* curPos) * (-0.0025))
    curPos += curVel
    # reward -= 1
    #get next state
    if curPos <= -1.2:
        curVel = 0
        curPos = -1.2
    # StateRewardVec.append((State[0],State[1],reward))
    return np.array([curPos, curVel])

def isFinished(State):
    if State[0] >= 0.6:
        return True
    return False

def getFeatureVector(S, A = 0):
    # d = S.shape[0] #Might be other one.
    curPos = (S[0] +1.2)/1.8
    curVel = (S[1] +0.07)/0.14
    #phi = np.array()
    phi = np.cos(np.pi * np.dot(c,[curPos,curVel]))
    return phi

def getReward(S):
    if isFinished(S):
        return 0
    return -1



#initialize starting state
S = np.array([-0.5,0.0]) #State
Sp = np.array([-0.5,0.0]) #Sp = S'
Q = 0 #Value(S,A)
Qold = 0 #Previous Q
curA = 2
n = 7 #Number of Fourier Harmonics (Size of Fourier basis)
d = S.size #Dimentionality of state space
numBasisFunc = (n+1)**d #Number of basis functions 
numActions = action.size #Number of basis functions
#get Fourier basis
ci = np.arange(0,n+1) #range [0,n]
cj = np.arange(0,n+1) #range [0,n]
c = []
for i in ci:
    for j in cj:
        c.append((i,j))
c = np.array(c)
#Initialize constants
lamda = 0.9 #Back propagation variable, lambda is special variable name
gamma = 0.1 #discount factor
#Setup learning rate so it decreases with Fourier Basis
alpha = np.zeros(shape=(c.shape[0])) #Learning rate
alpha[1] = 0.001 #Learning rate
for vec in range(1,alpha.size):
    alpha[vec] = alpha[1] / np.linalg.norm(c[vec,:])
alpha[0] = alpha[1]
z = np.zeros(shape=(numBasisFunc,numActions)) #Array for Lambda calculations
weights = np.zeros(shape=(numBasisFunc,numActions))
value = np.zeros(shape=(numActions))
phi = np.zeros(shape=(numBasisFunc))
phip = np.zeros_like(phi)
#Start Algorithm
#Initialize episode counter
epCounter = 0
rewardVec = []
numTrials = 10
numEpisodes = 100
for epCount in range(numTrials*numEpisodes):
    if epCount % numEpisodes == 0:
        epCounter = 0
    S[0] = np.random.uniform(-0.6,-0.4) #State
    S[1] = np.random.uniform(-0.01,0.01)
    Sp = S
    #Pick Action
    for A in action:
        value[A] = np.dot(phi,weights[:,A])
    curA = 2 #default action to forward
    if value[0] > value[1] and value[0] > value[2]:
        curA = 0
    elif value[1] > value[2]:
        curA = 1
    #Get Feature Vector
    phi = getFeatureVector(S)
    #Initialize Z to 0
    z = np.zeros(shape=(numBasisFunc,numActions)) #Array for Lambda calculations
    #initialize Qold
    Qold = 0
    reward = 0
    while (not isFinished(S)):
        #Store current S
        S = Sp
        #Take Action, get resulting state
        Sp = UpdateSimulation(S,curA)
        #Observe Reward, Update global reward
        R = getReward(Sp)
        reward += R
        #Pick Action
        for A in action:
            value[A] = np.dot(phi,weights[:,A])
        curA = 2 #default action to forward
        if value[0] > value[1] and value[0] > value[2]:
            curA = 0
        elif value[1] > value[2]:
            curA = 1
        #Plot Current State
        # if epCounter == numEpisodes-1:
        #     plt.cla()
        #     plt.ylim(-0.07,0.07)
        #     plt.xlim(-1.2,0.6)
        #     plt.plot(Sp[0], Sp[1], marker='o')
        #     plt.draw()
        #     plt.pause(0.001)
            StateRewardVec.append((Sp[0],Sp[1],value[curA]))
        
        
        #Store old feature Vector, get new feature vector
        phip = getFeatureVector(Sp)
        
        #Update Q, Qp
        Q = np.dot(weights[:,curA],phi)
        Qp = np.dot(weights[:,curA],phip)
        #Update delta
        delta = R + gamma * Qp - Q
        #Update Z
        z = gamma * lamda * z
        # z[:,curA] = gamma * lamda * z[:,curA] + (1-alpha*gamma*lamda*np.dot(z[:,curA],phi))*phi
        z[:,curA] = z[:,curA] + (1-alpha*gamma*lamda*np.dot(z[:,curA],phi))*phi
        #Update weights
        for A in action:
            weights[:,A] = weights[:,A] + alpha * (delta+Q-Qold)*z[:,A]
        weights[:,curA] -= alpha * (Q-Qold)*phi
        # weights[:,curA] += alpha * (delta + Q - Qold)*z[:,curA] - alpha * (Q-Qold)*phi
        #Store Q,phi,curA
        Qold = Qp
        phi = phip
        # curAp = curA
    epCounter += 1
    rewardVec.append((epCounter, -reward))
rewardVec = np.array(rewardVec)
StateRewardVec = np.array(StateRewardVec)
import pandas as pd
df = pd.DataFrame(data=rewardVec, columns=["Number_of_Episodes","Steps"])
# plt.figure()
# plt.plot(rewardVec[:,0], rewardVec[:,1])
sns.set_theme()
sns.relplot(data=df, kind="line", x="Number_of_Episodes",y="Steps").set(title='Convergence, 7 Harmonics')
# Plot Path
fig = plt.figure()
ax = plt.axes()
ax.set_title('Mountain Car Path')
ax.set_xlabel('Position')
ax.set_ylabel('Velocity')
plt.plot(StateRewardVec[:,0], StateRewardVec[:,1])

# #Plot 3D Path
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(StateRewardVec[:,0], StateRewardVec[:,1], StateRewardVec[:,2])
#Plot Value Function
ValueFunc = []
for pos in Position:
    for vel in Velocity:
        phi = getFeatureVector([pos,vel])
        tmp = np.zeros(shape=(numActions))
        for A in action:
            tmp[A] = np.dot(phi,weights[:,A])
        ValueFunc.append((pos, vel, max(tmp)))
ValueFunc = np.array(ValueFunc).reshape(len(Position),len(Velocity),3)
fig2 = plt.figure()
ax2 = plt.axes(projection='3d')
ax2.set_title('Inverse Value Function for 7 Fourier Harmonics and Path')
ax2.set_xlabel('Position')
ax2.set_ylabel('Velocity')
ax2.set_zlabel('Value')
ax2.plot_wireframe(ValueFunc[:,:,0], ValueFunc[:,:,1], -ValueFunc[:,:,2])
ax2.plot3D(StateRewardVec[:,0], StateRewardVec[:,1], -StateRewardVec[:,2])

# ValueFunc = np.array(ValueFunc).reshape(len(Position),len(Velocity),3)
# Velocity = np.linspace(-1,1,101)
# Position = np.linspace(-1,1,101)
# fig3 = plt.figure()
# ax3 = plt.axes(projection='3d')
# ax3.set_title('Basis [0,0]')
# ValueFunc = []
# for pos in Position:
#     for vel in Velocity:
#         phi = np.cos(np.pi * np.dot(c[0],[pos,vel]))
#         ValueFunc.append((pos, vel, phi))
# ValueFunc = np.array(ValueFunc).reshape(len(Position),len(Velocity),3)
# ax3.plot_wireframe(ValueFunc[:,:,0], ValueFunc[:,:,1], ValueFunc[:,:,2])
