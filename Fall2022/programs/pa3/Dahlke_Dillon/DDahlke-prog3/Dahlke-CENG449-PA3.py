#**********************************************************
# Dillon Dahlke
#
# CSC 449 - Programming Assignment 3 - Mountain Car
#
# Python Dependencies Needed:
#   gym
#   numpy
#   matplotlib
#
# Description: This program uses techniques to solve the
#   mountain car reinforcement learning problem.
#
#************************************************************
import gym
import numpy as np
import matplotlib.pyplot as plt

#Set up and build the environment
env = gym.make("MountainCar-v0", render_mode="rgb_array")
env.reset()

NUM_EPISODES = 1000
#How often to plot data points
SHOW_EVERY = 200

#Get the boundaries for Max and Min Cart position
MAX_POS = env.observation_space.high[0]
MIN_POS = env.observation_space.low[0]
#Get the boundaries for Max and Min cart velocity
MIN_VEL = env.observation_space.high[1]
MIN_VEL = env.observation_space.low[1]

#Number of actions in the Space
NUM_ACT = env.action_space.n

#Learning Rate
ALPHA = 0.2
#Discount Factor
GAMMA = 0.95


#Split up the observation space variables into buckets of 40
OBS_SIZE = [40] * len(env.observation_space.high)

OBS_RANGE = (env.observation_space.high - env.observation_space.low)
#Get the size for each discrete bin
BUCKET_SIZE = OBS_RANGE / OBS_SIZE
#Initialize the Q function
Qt = np.random.uniform(low=-2, high=0, size=(OBS_SIZE + [NUM_ACT]))
#Initialize variables for the plots
reward_per_ep = []
rewards_dict = {'ep': [], 'avg': [], 'min': [], 'max': []}

#Function to discretize continuous variables.
def to_discrete(state):
    ds = (state - env.observation_space.low) / BUCKET_SIZE
    return tuple(ds.astype(int))

def main():
    #Exploration Factor
    epsilon = 0.2
    #How fast to decay epsilon
    START_DECAY = 1
    END_DECAY = NUM_EPISODES
    DECAY_VAL = epsilon / (END_DECAY - START_DECAY)
    for episode in range(NUM_EPISODES):
        ep_reward = 0
        print(episode)
        state = env.reset()
        ds = to_discrete(state[0])
        complete = False
        truncated = False

        while not complete and not truncated:
            if np.random.random() > epsilon:
                action = np.argmax(Qt[ds]) 
            else:
                action = np.random.randint(0, NUM_ACT)
            next_state, reward, complete, truncated, _ = env.step(action)
            ep_reward += reward
            next_ds = to_discrete(next_state)
            #env.render()
            if not complete:
                #Current Q value (Uses offset) 
                Qval = Qt[ds + (action,)]
                next_Qval = np.max(Qt[next_ds])
                #Q Update
                Qval = (1-ALPHA) * Qval + ALPHA * (reward + (GAMMA*next_Qval))
                Qt[ds + (action,)] = Qval

            elif next_state[0] >= env.goal_position:
                Qt[ds + (action,)] = 0
                print(f"We made it on episode {episode}")
        if END_DECAY >= episode >= START_DECAY:
            epsilon -= DECAY_VAL
        
        reward_per_ep.append(ep_reward)

        if episode % SHOW_EVERY == 0:
            avg_r = sum(reward_per_ep[-SHOW_EVERY:])/len(reward_per_ep[-SHOW_EVERY:])
            rewards_dict['ep'].append(episode)
            rewards_dict['avg'].append(avg_r)
            rewards_dict['min'].append(min(reward_per_ep[-SHOW_EVERY:]))
            rewards_dict['max'].append(max(reward_per_ep[-SHOW_EVERY:]))
            
        

    env.close()

    #plt.plot(rewards_dict['ep'], rewards_dict['avg'], label='Average Reward')
    plt.plot(rewards_dict['ep'], rewards_dict['min'], label='Minimum Reward')
    plt.plot(rewards_dict['ep'], rewards_dict['max'], label='Maximum Reward')
    plt.legend(loc=4)
    plt.show()

if __name__ == "__main__":
    main()
        