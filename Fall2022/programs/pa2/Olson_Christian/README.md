# Programming Assignment 2: TD

Christian Olson

### Files:

1. my_agent.py - Trains agent
2. trained_agent.py - Runs trained agent from training.npy
3. my_agent_data.py - Settings for Agent
4. inverted_pendulum_server.py - simulation from Jonathan Mathews
5. inverted_pendulum.py - simulation form Jonathan Mathews
6. training.npy - stores values for trained agent, created when training
7. trained.npy - values used for trained agent

### Usage:

Training agent:

    $python my_agent.py

    $python inverted_pendulum_server <--animate>

Trained agent:

    $python trained_agent.py

    $python inverted_pendulum_server <--animate>

### Notes

The trained agent balances the pole until it nears a boundary. There are issues with getting the boundary reward
to spread amongst the values, so the agent struggles to handle itself near the boundaries.

# Installation

Jonathan Mathews

## Python server and example agent

Create a virtual environment to house dependencies for this project. This makes
it easy to remove all the dependencies later as the virtual environment can
just be deleted.

### Venv
If you want to use venv to make a virtual environment:

```
python3 -m venv /path/to/env
```

Activate the environment. You'll need to do this every time you open your terminal back up.

```
source /path/to/env/bin/activate
```

### Conda
If you want to use conda to make a virtual environment:

```
conda create -n env_name
```

To activate:

```
conda activate env_name
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
python inverted_pendulum_server.py --animate
```

To run as fast as possible, omit the animate argument

## Python example agent

```
python example_agent.py
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

