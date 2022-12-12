# inverted-pendulum-controller
Your assignment is to implement an inverted pendulum controller using tabular TD. Sarsa or
Q-learning should work for this problem. You can use Jonathan Mathews’ simulator, which can be
found on GitHub at https://github.com/ben-rambam/reinforcement_sim. Other
simulators are available online, or you can build your own simulator using the differential equa-
tions. The equations and approach can be found here https://www.cantorsparadise.com/modelling-and-simulation-of-inverted-pendulum-5ac423fed8ac.
Although the implementation of tabular TD is pretty straightforward, you will probably need
multiple trials to get the parameters tuned. If you wait until the last minute, you will not have
enough time to get it working well.



Alex Hanson

Python version: Python 3.9.0


Agent to balance a pole.
Agent is run on server which can be found here: https://github.com/ben-rambam/reinforcement_sim



## Requirements
Install the python dependencies.
> pip install -r requirements.txt

## Running
need to launch the server then launch the agent

### Inverted pendulum server
To run server
> python inverted_pendulum_server.py --animate

### Python agent
> python agent.py


## Running pre trained
> python agent.py -a -r -l trained_q




## Usage
usage: agent.py [-h] [-a] [-e EPISODES] [-s [QFILEOUT]] [-l QFILEIN] [-r]

description

optional arguments:
  -h, --help            show this help message and exit
  -a, --animate         Enables animation on server.
  -e EPISODES, --episodes EPISODES
                        Number of episodes to run.
  -s [QFILEOUT], --save [QFILEOUT]
                        File to save q values too. Defaults to 'q_values'
  -l QFILEIN, --load QFILEIN
                        The file to load q values from
  -r, --run             Keeps Constant epsilon and alpha of 0.1. When omited epsilon and alpha start at 0.99 and    
                        reduce to 0.1.






## Q-Learning
Loop for each episode:
Initialize S
Loop for each step of episode:
    Choose A from S using policy derived from Q (e.g., epsilon-greedy)
    Take action A, observe R, S`
    Q(S,A) <- Q(S,A) + alpha[R + gamma * max_a Q(S`, a) - Q(S,A)]
    S <- S`
until S is terminal


## Buckets
Because this is a continuous environment there is an infinite combination of state action pairs. Which would take an infinite amount of time to train. To deal with this we group similar states into groups called buckets.
