import math
import random
from re import I
import numpy
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
k = 7
features = (k+1)*(k+1) # (k+1)^n, n=2.   

def nextPosition(x, v):
    x += v
    if(x < -1.2):
        x = -1.2
    if(x > .5):
        x = .5 

    return x

def nextVelocity(x, v, a):
    v  = v + .001*a - .0025 * math.cos(3 * x)
    if(v < -.07):
        v = -.07
    if(v > .07):
        v = .07 
    if(x == -1.2):
        v = 0
    return v

def fourier(theta, i, j, x, xDot):
    return theta * math.cos( math.pi * ((i * x)+ (j*xDot))) 

def poly(theta, i, j, x, xDot):
    return theta * x**i * xDot**j



def gradient(theta, i, j, x, xDot):
    return -math.pi * theta * i  * math.sin(math.pi * ((i * x)+ (j*xDot))) - math.pi * j * math.sin(math.pi * ((i * x)+ (j*xDot)))

def gradientlog(theta, i, j, x, xDot):
    num1 = -math.pi * i  * math.sin(math.pi * ((i * x)+ (j*xDot))) 
    num2 = -math.pi * j * math.sin(math.pi * ((i * x)+ (j*xDot)))
    denom = math.cos( math.pi * ((i * x)+ (j*xDot))) 
    return ((num1 + num2)/denom/theta)

def polyGradient(theta, i, j, x, xDot):
    num1 = 0
    num2 = 0
    if(x != 0):
        num1 = i/x
    if(xDot != 0):
        num2 = j/xDot

    return num1+num2

y = numpy.zeros(1000) #for learning curve later

state = [0.0,0.0] #  position and velocity as doubles
nextState = [0.0,0.0]
state[0] = random.uniform(-.6, -.4)

w = numpy.zeros(features, numpy.double)

n=0
while(n < features):
    w[n] = random.uniform(0.001, 0.01)
    n += 1



episodes = 0
while(episodes < 1000):
    episodes += 1
    e = numpy.zeros(features, numpy.double) # each episode
    c = 0
    state[0] = random.uniform(-.6, -.4)
    state[1] = 0
    while(c < 1000):
        if(c == 0):
             v3 = 0.0
             i = 0
             j = 0
             while(i < k+1): # computes value of current state
                while(j < k+1):
                    v3 += poly(w[i*(k+1)+j], i, j, nextPosition(state[0], state[1]), nextVelocity(state[0], state[1], -1))
                    j += 1
                i += 1

        else:
            state[0] = nextState[0]
            state[1] = nextState[1]
    
       
    
        v0 = 0.0
        i = 0
        j = 0
        while(i < k+1): # computes value of next state and reverse throttle
            while(j < k+1):
                 v0 += poly(w[i*(k+1)+j], i, j, nextPosition(state[0], state[1]), nextVelocity(state[0], state[1], -1))
                 j += 1
            i += 1

        v1 = 0.0
        i = 0
        j = 0
        while(i < k+1): # computes value of next state and forward throttle
            while(j < k+1):
                 v0 += poly(w[i*(k+1)+j], i, j, nextPosition(state[0], state[1]), nextVelocity(state[0], state[1], 1))
                 j += 1
            i += 1


        if(v0 > v1):
            nextState[0] = nextPosition(state[0], state[1])
            nextState[1] = nextVelocity(state[0], state[1], -1)
        else:
            nextState[0] = nextPosition(state[0], state[1])
            nextState[1] = nextVelocity(state[0], state[1], 1)
        

        if(c==999):
            print(c)
            y[episodes-1] = c

        if(nextState[0] == .5): #terminal state
            delta = 0 - v3
            if(episodes%10 ==0):
                print(c)
                print(nextState[0], nextState[1])
            y[episodes-1] = c
            c = 1000
        else:
            delta = -1 + max(v0, v1) - v3
        v3 = max(v0, v1)

        i = 0
        j = 0

        while(i < k+1):
            while(j < k+1):
                e[i*(k+1)+j] = .9* e[i*(k+1)+j] + polyGradient(w[i*(k+1)+j], i, j, state[0], state[1])
                j += 1 
            i += 1

        n = 0 
        while(n < features):
            w[n] += .001*delta * e[n]
            n += 1


        
        c +=1
            


# learning curve

x = numpy.zeros(1000)

n = 0
while(n < 1000):
    x[n] = n 
    n += 1


         

plt.scatter(x, y)

plt.savefig('curvePoly.pdf')

#surface plot
fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')


x = numpy.arange(-1.2, .5, 0.05)
y = numpy.arange(-.07, .07, 0.05)

X, Y = numpy.meshgrid(x, y)

i = 0
j = 0

#Z = w[0] + w[1]*math.cos(math.pi * Y) + w[2]*math.cos(math.pi * 2 * Y)  + w[3]*math.cos(math.pi * 3 * Y)  + w[4]*math.cos(math.pi * X) + w[5]*math.cos(math.pi * Y + math.pi * X) + w[6]*math.cos(math.pi * 2 * Y + math.pi * X) + w[7]*math.cos(math.pi * 3 * Y + math.pi * X) + w[8]*math.cos(math.pi * 2 * X) + w[9]*math.cos(math.pi * Y +math.pi * 2 * X) + w[10]*math.cos(math.pi * 2 * Y +math.pi * 2 * X) + w[11]*math.cos(math.pi * 3 * Y +math.pi * 2 * X) + w[12]*math.cos(math.pi * 3 * X) + w[13]*math.cos(math.pi * Y +math.pi * 3 * X) + w[14]*math.cos(math.pi * 2 * Y +math.pi * 3 * X) + w[15]*math.cos(math.pi * 3 * Y +math.pi * 3 * X)


#surf = ax.plot_surface(X, Y, Z, cmap = plt.cm.cividis)

# Set axes label
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)

#fig.colorbar(surf, shrink=0.5, aspect=8)

#plt.show()