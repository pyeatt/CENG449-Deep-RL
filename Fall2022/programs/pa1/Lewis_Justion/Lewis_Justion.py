# Advanced Topics in Artifical Intelligence
# Justin Lewis
# 9/19/2022

BOARD_X = 4
BOARD_Y = 4
GAMMA = 0.95 # Gamme Value of 1 gives more accurate stochastic policy. 
             # Note the bottom left square -> you should be able to go up then left and get the same reward, but when 0.95, it only goes right. 
             # not sure if it's a result of something I did wrong in the program, but just an observation
THETA = 0.1

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
actions = [UP, RIGHT, DOWN, LEFT]

# for printing purposes
txt = {UP: "U", RIGHT: "R", DOWN: "D", LEFT: "L"}


def GenerateStates():
    # list of tuples with x,y
    states = []
    for y in range(BOARD_Y):
        for x in range(BOARD_X):
            states.append((x, y))
    return states

def PrintValues(v, deterministic=True):
    for s in GenerateStates():
        if type(v[s]) is not list: # print the values
            print("{:.2f}".format(v[s]), end="\n" if s[0] == 3 else "\t| ")
        else: # print the policy
            actions = v[s]
            if deterministic:
                actions = [actions[0]]
            print(''.join([txt[action] for action in actions]).center(4, " "), end="\n" if s[0] == 3 else " | ")


def GenerateValues(states): # Start with a (not) random value function. Called random in the book, is just zeros.
    values = {}
    for state in states:
        values[state] = 0
    return values


def sP(state, action): # sp for state prime, AKA where the agent would end up if they take an action. 
                       # For problems like the cleaning robot, if he did the cleaning action that had a chance of either going bakc
                       # to high or low battery after this, we would be able to pull in the pdf here. 
                       # handles the teleportation
    x = state[0] + (1 if action == RIGHT else -1 if action == LEFT else 0)
    y = state[1] + (1 if action == DOWN else -1 if action == UP else 0)
    x = max(0, x)
    x = min(BOARD_X - 1, x)
    y = max(0, y)
    y = min(BOARD_Y - 1, y)
    if state == (0, 2) and action == LEFT:
        return (BOARD_X - 1, BOARD_Y - 1)
    return (x, y)


def CalculateReward(state, action):  # prefill reward dictionary
    # teleportation action
    teleportState = BOARD_X * 2; # leftmost tile in 3rd row. 
    if state[1] * BOARD_X + state[0] == teleportState and action == LEFT:
        return -2
    # win state
    if (action in [RIGHT, DOWN]) and (state[0] == BOARD_X - 1 and state[1] == BOARD_Y - 1):
        return 0
    # defualt value
    if state == (3, 3):
        return 0
    return -1


def GenerateRewards(states): # generate all the rewards for states and actions
    # r(s, a)
    rewards = {}
    # may be simpler to just pregenerate with -1 in every spot, then
    # go in and modify the specific cases to the different values. 
    for state in states:
        for action in actions:
            rewards[state, action] = CalculateReward(state, action)
    return rewards


def pi(s, v): # policy
    bestActions = []
    statePrimes = [sP(s, a) for a in actions]
    maxValue = max([v[sp] for sp in statePrimes])

    for action in actions:
        sp = sP(s, action)
        # if sp == s: # bounced against edge
            # continue
        if v[sp] == maxValue:
            bestActions.append(action)
    if not bestActions: # idk how to handle, this shouldn't ever happen anyway unless something in the setup is messed up
        print("Scream and die")
        print(s)
        PrintValues(v)
        exit(1)
    return bestActions


def GeneratePolicy(states, v): # AKA policy improvement -> make it match the current value fxn
    policy = {}
    for state in states:
        policy[state] = pi(state, v)
    return policy

def pdf(a, s, policy): # not an actual probability distribution function, but does give the probability of taking action given current policy. 
    # I think stochastic-ness would be pulled in here
    # return probability of action given state
    if(a in policy[s]): # if greedy choice
        return 1/len(policy[s])
    return 0


def PolicyEvaluation(V, S, R, pi):
    stable = True
    originalValues = V.copy()
    while stable:
        delta = 0
        for s in S:
            v = V[s]
            _sum = 0
            for a in actions:
                r = R[s, a]
                sp = sP(s, a)
                # V[s] = sum_{s', r} p(s', r | s, policy(s)) [ r + gamma * V[s'] ]
                # sum += p(s', r | s, pi(s)) * other things  ----> p(s', r), not sure what the ", r" is for. pi(s) is just greedy -> whichever close value is highest
                #                                                          , So I think the probability distribution is where stochastic vs deterministic comes into play
                # sum += other things * [r + gamma * V[s']]  ----> r is the r(s, a) < aka r(s') >, gamma is a constant given in the assignment, and V[s'] is obvious. All of these are obvious actually
                _sum += pdf(a, s, pi) * (r + GAMMA * originalValues[sp])
            
        
            V[s] = _sum
            delta = max(delta, abs(v - _sum))
        pi = GeneratePolicy(S, V)
        if(delta < THETA): # this theta thing was listed in the book, but I don't think it's necessary. 
            stable = False
    return V, pi # return new values from value iteration



if __name__ == "__main__":
    # 1. initialization
    states = GenerateStates()
    rewards = GenerateRewards(states)
    values = GenerateValues(states)
    policy = GeneratePolicy(states, values)
    stable = False

    #2. Policy Evaluation (AKA Value Iteration ? )
    # values changed in the thingy
    while not stable:
        values, newPolicy = PolicyEvaluation(values.copy(), states, rewards, policy.copy())
        if policy == newPolicy:
            stable = True
        policy = newPolicy
    print("DETERMINISTIC POLICY")
    PrintValues(policy) # literally same policy, just only printing one so that
    # it looks like it's deterministic because it has no bearing on calculations -> it still follows the bellman equation.
    print("DETERMINISTIC VALUES")
    PrintValues(values)


    print("\nSTOCHASTIC POLICY")
    PrintValues(policy, deterministic=False)
    print("STOCHASTIC VALUES")
    PrintValues(values) # literally same values

    init = 1
    for p in policy.values():
        init *= len(p)
    print("\nNUMBER OF OPTIMAL DETERMINISTIC POLICIES")
    print(init)

