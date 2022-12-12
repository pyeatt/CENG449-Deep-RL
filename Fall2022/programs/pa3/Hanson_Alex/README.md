Alex Hanson


# mountain-car

CSC 449/549 — Advanced Topics in Artificial Intelligence
Deep Reinforcement Learning
Fall, 2022
Programming Assignment 3
Implement Sarsa(λ) for the Mountain Car problem as described in Sutton and Barto. Use linear
function approximation with Fourier basis functions.
https://people.cs.umass.edu/~pthomas/papers/Konidaris2011a.pdf

Write a report, and include at lesat the following:
1. Show learning curves for order 3, 5, and 7 Fourier bases, for a fixed setting of alpha and epsilon, and
gamma = 1, lambda = 0.9.
2. Create a surface plot of the value function (the negative of the value function) of the learned
policies after 1, 000 episodes, for the above orders. (Hint: Your plot should look like the one
in Sutton and Barto, but smoother.)
3. The Mountain Car contains a negative step reward and a zero goal reward. What would
happen if gamma was less than 1 and the solution was many steps long? What would happen if
we had a zero step cost and a positive goal reward, for the case where gamma = 1, and the case
where gamma < 1?

## Requirements
python

### packages
matplotlib
numpy


## Files
README.md - this file.
main.py - runs simulation.
mountain_car_simulator.py - the mountain car environment.
render.py - contains a function to render position vector to an animation.
color_text.py - collection of functions to print with ascii color codes.
out_3.gif - simulation of order 3 Fourier base after 1000 episodes of training.
out_5.gif - simulation of order 5 Fourier base after 1000 episodes of training.
out_7.gif - simulation of order 7 Fourier base after 1000 episodes of training.


## Usage
usage: main.py [-h] [--part_1] [--part_2]

optional arguments:
  -h, --help  show this help message and exit
  --part_1    Generate graphs for part 1. Learning Curves 3,5,7
  --part_2    Generate surface for part 2. Value function 3,5,7



## Simulation of Mountain car
### State variables
Velocity = (-0.07, 0.07)
Position = (-1.2, 0.6)
### Actions
motor = (left, neutral, right)
### Reward
For each time step:
reward = -1
### Update function
For each time step:
Action = [-1, 0, 1]
Velocity = Velocity + (Action) * 0.001 + cos(3 * Position) * (-0.0025)
Position = Position + Velocity
### Starting condition
Position = -0.5
Velocity = 0.0
### Termination condition
Simulation ends when:
Position >= 0.6