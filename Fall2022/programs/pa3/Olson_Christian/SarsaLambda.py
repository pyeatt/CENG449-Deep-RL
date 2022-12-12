import numpy as np
import time
from MountainCar import MountainCar
from FourierBasis import FourierBasis

def SarsaLambda(sim, F, numEpisodes = 1000):
    n = F.getN()
    params = np.zeros(n)
    traces = np.zeros(n)
    stepsize = 0.001
    discount = 1
    traceDecayRate = 0.9
    expRate = 0
    reward = 0

    # record steps
    episodeSteps = []
    step = 0

    # for each episode
    for i in range(numEpisodes):
        print("Epidsode: ", i)
        start = time.time()

        # initialize state and get initial action
        sim.reset()
        state = sim.getState()
        action = chooseAction(sim, F, params, state, expRate)

        # get initial features, zero traces, and zero old Q
        features = F.features(MountainCar.normState(state), MountainCar.normAction(action))
        traces = np.zeros(n)
        oldQ = 0

        # step tracking
        step = 0

        while not sim.isFinished():
            # get reward, new state, new action, and new features
            reward = sim.update(action)
            newState = sim.getState()
            newAction = chooseAction(sim, F, params, newState, expRate)
            newFeatures = F.features(MountainCar.normState(newState), MountainCar.normAction(newAction))

            # calculate current and nest Q
            Q = np.dot(params, features)
            newQ = np.dot(params, newFeatures)

            # calculate update: R + discount * Q(S',A') - Q(S,A)
            error = reward + discount * newQ - Q

            # increment traces: z = discount * decay * z + (1 - step*discount*decay*z*x)x
            traces *= (discount * traceDecayRate)
            traces += ((1 - stepsize * discount * traceDecayRate * np.dot(traces, features)) * features)
            params += (stepsize * (error + Q - oldQ) * traces)
            params -= (stepsize * (Q - oldQ) * features)

            # change current state to new state
            oldQ = newQ
            features = newFeatures
            action = newAction
            state = newState

            # increment step
            step += 1

        # record epidsode
        episodeSteps.append(step)
        print("Duration: ", time.time() - start, "\tSteps: ", step)

    return params, episodeSteps

def chooseAction(sim, F, params, state, e):
    p = np.random.random(1)[0]

    # random action
    if p < e:
        j = np.random.randint(0, 3, 1)[0] - 1 # Actions are -1, 0, 1
        return j

    # get normalized actions
    af = MountainCar.normAction(sim.FORWARD)
    ai = MountainCar.normAction(sim.IDLE)
    ab = MountainCar.normAction(sim.BACKWARD)

    # normalize state
    state = MountainCar.normState(state)

    # get values
    Q = np.array([F.Q(params, state, ab), F.Q(params, state, ai), F.Q(params, state, af)])

    # best action
    i = np.argmax(Q) - 1    # Actions are -1, 0, 1
    return i
