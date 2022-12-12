import mountain_car_sim as mcs
import fourier_basis as fb
import numpy as np
import random

# if action == None, then evaluate current state without taking an action. otherwise,
#  action should be one of [-1, 0, 1]
def getQuality(state: mcs.MCSim, weights: np.ndarray, features: fb.FourierBasis, action: int = None) -> float:
    if (np.isnan(weights).any() or np.isinf(weights).any()):
        raise ValueError("Invalid value found in weights array: " + str(weights))

    # note that this doesn't actually update with a new state, newState is just
    #  for this calculation
    newState = None
    if (action != None):
        newState = state.stepNoUpdate(action)
    else:
        newState = state

    featuresEval = features.calculate(newState)

    quality = weights @ featuresEval

    return quality

def chooseBestAction(state: mcs.MCSim, weights: np.ndarray, features: fb.FourierBasis) -> int:
    actions = np.array([-1, 0, 1])
    qualities = np.empty(3)

    for i in range(len(actions)):
        qualities[i] = getQuality(state, weights, features, actions[i])

    bestActionIndex = np.argmax(qualities)

    return actions[bestActionIndex]

def chooseActionEGreedy(state: mcs.MCSim, weights: np.ndarray, features: fb.FourierBasis, epsilon: float = 0.1) -> int:
    if (epsilon != 0 and random.random() <= epsilon):
        action = int(random.random() * 3) - 1
        return action
    else:
        return chooseBestAction(state, weights, features)

