import numpy as np
import gym
import matplotlib.pyplot as plt

posSpace = np.linspace(-1.2, 0.6, 20)
velSpace = np.linspace(-0.07, 0.07, 20)

def getState(obs):
    pos, vel = obs
    posBin = np.digitize(pos, posSpace)
    velBin = np.digitize(vel, velSpace)

    return (posBin, velBin)

def maxAction(Q, state, actions=[0,1,2]):
    values = np.array([Q[state,a]for a in actions])
    action = np.argmax(values)

    return action





if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000
    numGames = 10000
    alpha = 0.1
    gamma = 0.99
    eps = 1.0

    states = []
    for pos in range(21):
        for vel in range(21):
            states.append((pos, vel))

    Q = {}    
    for state in states:
        for action in [0,1,2]:
            Q[state, action] = 0


    for i in range(numGames):
        done = False
        obs = env.reset()
        state = getState(obs)
        if i % 1000 == 0 and i > 0:
            print('episode ', i, 'score ', score, 'epsilon %.3f' % eps)
        score = 0
        totalReward = np.zeros(numGames)
#        print(obs, state)
        while not done:
            action = np.random.choice([0,1,2]) if np.random.random() < eps \
                else maxAction(Q, state)
            obs_, reward, done, info = env.step(action)
            state_ = getState(obs_)
#            print(obs_, state_)
#            input()
            score += reward
            action_ = maxAction(Q, state_)
            Q[state, action] = Q[state, action] + \
                alpha*(reward + gamma*Q[state_, action_] - Q[state, action])
            state = state_
        totalReward[i] = score
        eps = eps - 2/numGames if eps > 0.01 else 0.01
    
    mean_rewards = np.zeros(numGames)
    for r in range(numGames):
            mean_rewards[r] = np.mean(totalReward[max(0, r-50):(r+1)])
    plt.plot(mean_rewards)
    plt.savefig('myCar.png')

