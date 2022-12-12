Welcome to inverted pendulum world!

This program is for balancing an inverted pendulum using Q-learning
temporal differencing off-policy algorithm. 

To run the program, please follow the following steps. These steps assume
you already have all the dependencies installed. 

1) First, run the inverted_pendulum_server.py from this folder. I changed
   the starting state to be x=0, x_dot=0, theta=0, theta_dot=0. This change
   is important for my code to work. 

2) Then run my_agent.py. Make sure the .npy file is in the same directory 
   as my_agent.py.

The output should be a balancing pole, and number of attempts/tries. 