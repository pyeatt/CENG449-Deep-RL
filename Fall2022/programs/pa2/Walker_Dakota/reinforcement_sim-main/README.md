*******************************************************************************
This model needs furter training to be more accurate. It currently only has 
33,000 recorded episodes completed. While it can successfully balance the pole, 
there is still a lot of state-space left undiscovered. Because I chose to 
initialize the Q-Table with zeros, it always wants to try out those new actions 
in every state.

I ended up settling with a crude state-space of 15 bins per state variable, 
with 7 possible actions. While crude, the state-space does include 360 degrees
of pole travel which has seriously slowed down the convergance time. The 7 
actions are 3 different strengths in both directions, and a push of zero force. 
The three strengths are 4, 8, and 12. The rewards are -1 for dipping below
horizontal, and -5 for hitting the wall. The trained episodes used a value of
-1 for the walls.

The only special instructions are ensure you are in the directory that contains 
the program and q_table.npy file, and then run q_agent.py. The rest is compiled 
and  ran the same as the example agent provided.

- Dakota Walker, 10/10/2022
*******************************************************************************

# Installation

## Python server and example agent

Create a virtual environment to house dependencies for this project. This makes
it easy to remove all the dependencies later as the virtual environment can
just be deleted.

```
python3 -m venv /path/to/env
```

Activate the environment. You'll need to do this every time you open your terminal back up.

```
source /path/to/env/bin/activate
```

Install the python dependencies.

```
pip install -r requirements.txt
```

## C++ example agent

Follow the [cppzmq installation
instructions.](https://github.com/zeromq/cppzmq#build-instructions) then build
the example agent as follows:

```
mkdir build
cd build
cmake ..
make
```

# Running

You will need to start the server and a single agent in two separate terminals
so they are both running at the same time. Also, if one dies you will have to
close and restart the other one as well or it won't reconnect.

## Inverted pendulum server

To run the server with animation of the pendulum enabled

```
./inverted_pendulum_server.py --animate
```

To run as fast as possible, omit the animate argument

## Python example agent

```
./example_agent.py
```

## C++ example agent

If you've made changes to the agent, recompile it:

```
cd build
make
cd ..
```

Then run the agent with this command:

```
./build/example_agent_cpp
```

