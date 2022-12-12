#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#np.set_printoptions(precision=2, suppress=False)
pd.set_option('display.float_format', '{:.2g}'.format)

DOWN = 0
LEFT = 1
UP = 2
RIGHT = 3
actions = [DOWN, LEFT, UP, RIGHT]

GRID_SIZE = 4
states = np.arange(GRID_SIZE**2, dtype=int)


def r_func(state, action, final_state):
    """Computes the reward for a given state, action, and final state

    Parameters
    ----------
    state : int
        The starting state 
    action : int
        The action taken
    final_state : int
        The final state
    """
    r = 0
    if state == final_state == 15:
        r += 0
    elif state == 8 and final_state == 15:
        r += -2
    else:
        r += -1
    return r


def p_func(final_state, action, state):
    """Computes the conditional probability p(s' | s, a)

    This is the probability that we end up in state `final_state` 
    given that we started in state `state` and took action `action`.

    The world consists of a grid with the states numberd as shown below.

    +----+----+----+----+
    |  0 |  1 |  2 |  3 |
    +----+----+----+----+
    |  4 |  5 |  6 |  7 |
    +----+----+----+----+
    |  8 |  9 | 10 | 11 |
    +----+----+----+----+
    | 12 | 13 | 14 | 15 |
    +----+----+----+----+

    The available actions are to move DOWN, LEFT, UP, and RIGHT
    
    * Attempting to move left from state 8 takes you directly to state 
      15 with a reward of -2.
    * Attempting to move right or down from state 15 takes you back to 
      state 15 with a reward of 0.
    * Attempting any other motion that takes you outside the grid returns 
      you to your starting state with a reward of -1.
    * Any other motion takes you in the expected direction with a score of -1.

    Parameters
    ----------
    state : int
        The starting state
    action : int
        The action taken
    final_state : int
        The final state

    Raises
    ------
    Exception
        The agent attempted an invalid action
    """

    global GRID_SIZE
    x = state % GRID_SIZE
    y = state // GRID_SIZE
    xprime = final_state % GRID_SIZE
    yprime = final_state // GRID_SIZE

    if action == DOWN:
        xnew = x
        ynew = min(y+1, GRID_SIZE-1)
    elif action == LEFT:
        if state == 8:
            xnew = 3
            ynew = 3
        else:
            xnew = max(x-1, 0)
            ynew = y
    elif action == UP:
        xnew = x
        ynew = max(y-1, 0)
    elif action == RIGHT:
        xnew = min(x+1, GRID_SIZE-1)
        ynew = y
    else:
        raise Exception("Action {} isn't valid".format(action))

    if xprime == xnew and yprime == ynew:
        prob = 1
    else:
        prob = 0

    return prob


def pi_0(a, s):
    """A starting policy that weights each action equally

    Parameters
    ----------
    a : int
        The action taken
    s : int
        The starting state
    """
    if a == DOWN:
        pi = 0.25
    elif a == LEFT:
        pi = 0.25
    elif a == UP:
        pi = 0.25
    elif a == RIGHT:
        pi = 0.25

    return pi


def evaluate_policy(pi, values):
    """A method to evaluate 1 timestep of a policy
    """
    new_values = np.zeros_like(values)
    for s in states:
        for a in actions:
            action_prob = pi(a, s)
            for sprime in states:
                final_state_prob = p_func(sprime, a, s)
                r = r_func(s, a, sprime)
                val = action_prob * final_state_prob * \
                    (r + gamma*values[sprime])

                new_values[s] += val
    return new_values


def eval_policy_grid(pi, values):
    """A faster method to evaluate 1 timestep of a policy
    """
    G = r+gamma*values
    new_values = np.einsum('as,asp,asp->s', pi, p, G)

    return new_values


# Precompute arrays to hold our policy, state-transition probabilities
# and rewards. 
pi = np.array([[pi_0(a, s) for s in states] for a in actions])
p = np.array([[[p_func(sp, a, s) for sp in states]
             for s in states] for a in actions])
r = np.array([[[r_func(s, a, sp) for sp in states]
             for s in states] for a in actions])

# Run a couple tests to validate method
# It's important that sprime is the final axis for broadcasting to work nicely
a = 1
s = 8
sp = 15
assert pi[a, s] == 0.25, "pi indexing is wrong"
assert p[a, s, sp] == 1, "p indexing is wrong"
assert r[a, s, sp] == -2, "r indexing is wrong"

gamma = 0.95
v_rand = np.random.uniform(-20.0, 20.0, len(states))
eval1 = evaluate_policy(pi_0, v_rand)  # easier to code but slow
eval2 = eval_policy_grid(pi, v_rand)  # easier to mess up but way faster
assert np.max(np.abs(eval1-eval2)
              ) < 0.00000001, "The methods don't agree: \n{} \n!= \n{}".format(eval1, eval2)

def iterate_v(pi, v):
    """A function to repeatedtly evaluate a policy until convergence

    Parameters
    ----------
    pi : np.ndarray()
        A 2D array indexed with [action, state]
    v : np.ndarray()
        A 1D array with the initial values for each state
    """
    diff = 1000
    count = 0
    while diff > 0.001:
        vprime = eval_policy_grid(pi, v)
        diff = np.max(np.abs(v-vprime))
        v = vprime
        count += 1
    return vprime, count


def pick_best_policy_deterministic(v):
    G = r+gamma*v

    q = np.einsum('asp,asp->as', p, G)

    best_action = np.argmax(q, axis=0)
    pi = np.zeros((len(actions), len(states)))
    pi[best_action, states] = 1
    return pi


def pick_best_policy_stochastic(v):
    G = r+gamma*v

    q = np.einsum('asp,asp->as', p, G)

    best_action = np.argmax(q, axis=0)
    best_qs = q[best_action, states]
    EQUIV_POLICY_THRESH = 0.001
    prob_q = (best_qs-q) <= EQUIV_POLICY_THRESH
    sum_q = np.sum(prob_q, axis=0)

    num_equiv_policies = np.prod(sum_q)

    return prob_q/sum_q, num_equiv_policies


# Uncomment the following if you want to do Policy Iteration. 
# It's much slower and can't handle a gamma of 1 which means we 
# may not find the best policy

"""
gamma = 0.999 # gamma less than 0.999 won't find the best policy 
v = 0 * np.ones_like(states, dtype=float)
vprime = np.zeros_like(v)
piprime = pi


policy_iter_count = 0
value_iter_count = 0
policy_changed = True
while policy_changed:
    policy_iter_count += 1
    pi = piprime
    v = vprime
    vprime, iter_count = iterate_v(pi, v)
    value_iter_count += iter_count
    piprime = pick_best_policy_deterministic(vprime)
    if np.all(pi == piprime):
        policy_changed = False
print("\n\n")
print("Policy iteration completed in {} policy iterations and {} value iterations".format(policy_iter_count, value_iter_count))
print("Optimal values : ")
print(v.reshape((GRID_SIZE, GRID_SIZE)))
print("Optimal deterministic policy : ")
print(pi)
pi, num_policies = pick_best_policy_stochastic(v)
print("Optimal stochastic policy:")
print(pi)
print("There are {} deterministic policies".format(num_policies))
"""

################################################################################
###        a gamma less than one may result in a non-optimal solution        ###
################################################################################
gamma = 0.95
v = 0 * np.ones_like(states, dtype=float)
vprime = 0 * np.ones_like(v)

count = 0
diff = 1000
while diff > 0.000001:
    count += 1
    v = vprime
    G = r+gamma*v
    q = np.einsum('asp,asp->as', p, G)
    vprime = np.max(q, axis=0)
    diff = np.max(np.abs(vprime-v))

print("\n\n")
print("Optimizing for gamma = {}".format(gamma))
print("Value iteration completed in {} iterations".format(count))
print("Optimal values : ")
print(v.reshape((GRID_SIZE, GRID_SIZE)))

pi = pick_best_policy_deterministic(v)
df = pd.DataFrame({"Down":pi[0],"Left":pi[1], "Up":pi[2], "Right":pi[3]})
print("One optimal deterministic policy:")
print(df)
pi, num_policies = pick_best_policy_stochastic(v)
df = pd.DataFrame({"Down":pi[0],"Left":pi[1], "Up":pi[2], "Right":pi[3]})
print("Optimal stochastic policy:")
print(df)
print("\nThere are {} optimal deterministic policies for gamma = {}".format(num_policies, gamma))

fig, [ax1, ax2] = plt.subplots(1,2)
fig.suptitle("Optimum values and policy for $\gamma={}$".format(gamma))
v_square = v.reshape((GRID_SIZE, GRID_SIZE))
ax1.set_title("Optimal values")
ax1.set_xticks([])
ax1.set_yticks([])
ax1.imshow(v_square)
for (j, i), label in np.ndenumerate(v_square):
    ax1.text(i, j, "{:.3f}".format(label), color='black', ha='center', va='center')

ax2.set_title("Optimal policy")
ax2.set_xticks([])
ax2.set_yticks([])
ax2.imshow(v_square)
pi_square = pi.reshape( (-1, GRID_SIZE, GRID_SIZE ))
for (action, j, i), action_prob in np.ndenumerate(pi_square):
    arrow_length = 0.3
    if action_prob > 0.000001:
        if action == DOWN:
            ax2.quiver( i,j, 0, -arrow_length, color='black', units='xy',scale=1)
        elif action == LEFT:
            ax2.quiver( i,j, -arrow_length, 0, color='black', units='xy',scale=1)
        elif action == UP:
            ax2.quiver( i,j, 0, arrow_length, color='black', units='xy',scale=1)
        elif action == RIGHT:
            ax2.quiver( i,j, arrow_length, 0, color='black', units='xy',scale=1)
gamma = 1.0
v = 0 * np.ones_like(states, dtype=float)
vprime = 0 * np.ones_like(v)

count = 0
diff = 1000
while diff > 0.000001:
    count += 1
    v = vprime
    G = r+gamma*v
    q = np.einsum('asp,asp->as', p, G)
    vprime = np.max(q, axis=0)
    diff = np.max(np.abs(vprime-v))

print("\n\n")
print("Optimizing for gamma = {}".format(gamma))
print("Value iteration completed in {} iterations".format(count))
print("Optimal values : ")
print(v.reshape((GRID_SIZE, GRID_SIZE)))

pi = pick_best_policy_deterministic(v)
df = pd.DataFrame({"Down":pi[0],"Left":pi[1], "Up":pi[2], "Right":pi[3]})
print("One optimal deterministic policy:")
print(df)
pi, num_policies = pick_best_policy_stochastic(v)
df = pd.DataFrame({"Down":pi[0],"Left":pi[1], "Up":pi[2], "Right":pi[3]})
print("Optimal stochastic policy:")
print(df)
print("\nThere are {} optimal deterministic policies for gamma = {}".format(num_policies, gamma))

fig, [ax1, ax2] = plt.subplots(1,2)
fig.suptitle("Optimum values and policy for $\gamma={}$".format(gamma))
v_square = v.reshape((GRID_SIZE, GRID_SIZE))
ax1.set_title("Optimal values")
ax1.set_xticks([])
ax1.set_yticks([])
ax1.imshow(v_square)
for (j, i), label in np.ndenumerate(v_square):
    ax1.text(i, j, "{:.3f}".format(label), color='black', ha='center', va='center')

ax2.set_title("Optimal policy")
ax2.set_xticks([])
ax2.set_yticks([])
ax2.imshow(v_square)
pi_square = pi.reshape( (-1, GRID_SIZE, GRID_SIZE ))
for (action, j, i), action_prob in np.ndenumerate(pi_square):
    arrow_length = 0.3
    if action_prob > 0.000001:
        if action == DOWN:
            ax2.quiver( i,j, 0, -arrow_length, color='black', units='xy',scale=1)
        elif action == LEFT:
            ax2.quiver( i,j, -arrow_length, 0, color='black', units='xy',scale=1)
        elif action == UP:
            ax2.quiver( i,j, 0, arrow_length, color='black', units='xy',scale=1)
        elif action == RIGHT:
            ax2.quiver( i,j, arrow_length, 0, color='black', units='xy',scale=1)


plt.show(block=True)
