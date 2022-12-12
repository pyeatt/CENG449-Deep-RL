# Installation

## Python server and example agent

Create a virtual environment to house dependencies for this project. This makes
it easy to remove all the dependencies later as the virtual environment can
just be deleted.

```
python3 -m venv /path/to/env
```

Activate the environment.

```
source /path/to/env/bin/activate
```

Install the python dependencies.

```
pip install -r requirements.txt
```

# Running

You will need to start the server and agent in two separate terminals,
so they are both running at the same time.

## Inverted pendulum server

To run the server with animation of the pendulum enabled

```
./inverted_pendulum_server.py --animate
```


## Python example agent

```
./agent.py
```

## Project Details 
I decided to implement Q-Learning since in theory it will converge faster for a problem like this.

I settled on an alpha of 0.85 since that has been show to produce the best results in Q-Learning. I used multiple values 
for my epsilon, but most of the time I used 0.8 since it seemed to give me the best results. 

By default, the agent will start a demo where it will randomly generate a start x-position, x-velocity, 
theta-position, and theta-velocity. (these values are bounded to keep the start near the center)

The training function is simply called train. I completed about 2.5 million training cycles 
in an attempt to cover most of the state space 
(each cycle started in a randomly generated position in an attempt to cover lots of my state space). 
The balancing is fairly stable, but I could have reduced training time by simplifying my state-space.





