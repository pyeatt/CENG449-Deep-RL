import gym
import numpy as np
import matplotlib.pyplot as plt

num_episodes = 1000

# Hyper Parameters
alpha = 0.001   # learning rate
epsilon = 0.0   # chance of picking random action (e-greedy)
gamma = 1       # trace decay
lamda = 0.9     # discount factor (lambda is keyword)
order = 3  # Fourier Order


env = gym.make('MountainCar-v0')
num_actions = 3  # specific to this environment
features = np.shape(env.observation_space)[0]
'''
## Observation Space
    The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:
    | Num | Observation                          | Min   | Max  | Unit         |
    |-----|--------------------------------------|-------|------|--------------|
    | 0   | position of the car along the x-axis | -1.2  | 0.6  | position (m) |
    | 1   | velocity of the car                  | -0.07 | 0.07 | velocity (v) |

## Action Space
    There are 3 discrete deterministic actions:
    - 0: Accelerate to the left
    - 1: Don't accelerate
    - 2: Accelerate to the right
'''


def createFourierBasis(order, features):
    temp = tuple([np.arange(order + 1)] * features)
    basis = np.array(np.meshgrid(*temp)).T.reshape(-1, features)
    return basis


# normalized value = (initial - min)/(max - min) in a range from 0-1
def normalize(state):
    norm_state = np.empty(np.shape(state))
    for i in range(np.shape(state)[0]):
        env_min = env.observation_space.low[i]
        env_max = env.observation_space.high[i]
        norm_state[i] = (state[i] - env_min) / (env_max - env_min)

    return norm_state


def computeFourierBasis(state, basis):
    state = normalize(state)
    return np.cos(np.pi * np.dot(basis, state))


# choose best action for the given state
def best_action(state, weights, basis):
    temp = np.zeros([1, num_actions])
    # compute value of all actions in the state
    for i in range(num_actions):
        temp[0, i] = np.dot(weights[:, i], computeFourierBasis(state, basis))
    # return index to best action
    return np.argmax(temp)


# choose best action for the given state, but returns the calculated value of the action
def best_value(state, weights, basis):
    temp = np.zeros([1, num_actions])
    # compute value of all actions in the state
    for i in range(num_actions):
        temp[0, i] = np.dot(weights[:, i], computeFourierBasis(state, basis))

    # print("State:", state[0],state[1],': ', temp[0, 0], temp[0, 1], temp[0, 2])
    return np.max(temp)


def e_greedy(state, epsilon, weights, basis):
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()  # randomly select action
    else:
        # pick best action for the current state
        action = best_action(state, weights, basis)
    return action


data_steps = []
data_episodes = []
sum = 0
worst = 0
best = 99999
basis = createFourierBasis(order, features)
weights = np.zeros([(order+1)**features, num_actions])  # (features x actions)
for i in range(0,num_episodes):
    # play last 10 episodes visually
    if i > num_episodes - 10:
        env = gym.make('MountainCar-v0', render_mode="human")
        env.reset()
    done = False
    step = 0
    state, info = env.reset()
    traces = np.zeros(np.shape(weights))  # traces
    action = e_greedy(state, epsilon, weights, basis)  # epsilon greedy selection
    # noinspection GrazieInspection
    while not done:
        step += 1
        # Take action, observe R, S'
        n_state, reward, terminated, truncated, info = env.step(action)
        # Calc TD error
        delta = reward - np.dot(weights[:, action], computeFourierBasis(state, basis))
        # Pick next action
        n_action = e_greedy(n_state, epsilon, weights, basis)

        delta = delta + gamma * np.dot(weights[:, n_action], computeFourierBasis(n_state, basis))

        traces[:, action] += computeFourierBasis(state, basis)  # accumulate traces
        weights = weights + delta * alpha * traces  # update the weight vector

        if not terminated:
            traces = traces * gamma * lamda  # decay traces

            state = n_state
            action = n_action
        done = terminated

    # Uncomment to print out episode finished message
    # print("Episode ", i, " finished after ", step, " time steps")
    sum += step
    if step > worst:
        worst = step
    if step < best:
        best = step
    if i % 10 == 0:
        data_steps.append(step)
        data_episodes.append(i)

print("Run Average:", sum/num_episodes)
print("Best:", best, " Worst:", worst)
env.close()

# Plot learning curve
plt.plot(data_episodes, data_steps)
plt.title(f'Learning Curve: Steps per Episode. Fourier O({order})')
plt.xlabel('Episodes')
plt.ylabel('Steps')
plt.axis([-20, 1000, 0, 2000])
plt.xticks(range(0, 1001, 100))
plt.yticks(range(0, 2001, 200))
plt.show()

# Plot value function surface
units = 200
pos = np.linspace(env.observation_space.low[0], env.observation_space.high[0], units)
vel = np.linspace(env.observation_space.low[1], env.observation_space.high[1], units)
z = np.empty([units, units])

for i in range(units):
    for j in range(units):
        state = [pos[i], vel[j]]
        z[i][j] = -best_value(state, weights, basis)

x, y = np.meshgrid(pos, vel)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Plot models:
p = ax.plot_surface(x, y, z, cmap="plasma")

# Plot settings:
ax.set_title(f'Value Function Surface for Fourier O({order})')
ax.set_xlim3d(np.min(x), np.max(x))
ax.set_ylim3d(np.min(y), np.max(y))
ax.set_xlabel('position')
ax.set_ylabel('velocity')
ax.set_zlabel('steps to go')
fig.colorbar(p, pad=0.2)

ax.elev = 45
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Plot models:
p = ax.plot_surface(x, y, z, cmap="plasma")

# Plot settings:
ax.set_title(f'Value Function Surface for Fourier O({order})')
ax.set_xlim3d(np.min(x), np.max(x))
ax.set_ylim3d(np.min(y), np.max(y))
ax.set_xlabel('position')
ax.set_ylabel('velocity')
ax.set_zticks([])
fig.colorbar(p, pad=0.01)

ax.elev = 90
plt.show()