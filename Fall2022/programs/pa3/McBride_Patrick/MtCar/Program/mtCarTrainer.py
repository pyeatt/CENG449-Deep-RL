import gym
import numpy as np
from sarsaLambda import SarsaLambda
from PlottingHelpers import plotResults
from PlottingHelpers import plotQvals

if __name__ == "__main__":
    makePlots = False        # Choose if graphs are generated
    numOfRenderedRuns = 3   # Number of runs to render after training is finished
    episodes = 1000         # Number of episodes to run
    basis = "Fourier"       # Set to "Radial" or "Fourier"
    order = 3               # Set the order of the basis

    alpha = 0.001
    gamma = 1.0
    lamb = 0.9
    epsilon = 0.0

    env = gym.make('MountainCar-v0')  # make environment

    stepsPerEp = np.zeros([episodes])  # initialize array to store run data

    agent = SarsaLambda(order=order, dimension=env.observation_space.shape[0], actions=env.action_space.n, basis=basis
                        , gamma=gamma, lamb=lamb, epsilon=epsilon, alpha=alpha)  # create agent

    # Train the agent
    for e in range(0, episodes):
        agent.wipeTraces()
        done = False
        stepsTaken = 0

        if e == episodes - numOfRenderedRuns:  # render selected amount of trained episodes
            env = gym.make('MountainCar-v0', render_mode='human')

        # Set initial state
        state = ((env.reset()[0] - env.observation_space.low) / (
                env.observation_space.high - env.observation_space.low))

        action = agent.nextAction(state)  # Choose first action

        while not done:  # Run episode
            splus, reward, done, legacy, info = env.step(action)
            splus = (splus - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
            aplus = agent.nextAction(splus)
            agent.update(state, action, reward, splus, aplus, terminal=done)
            state = splus
            action = aplus
            stepsTaken += 1

        print("Episode: " + str(e + 1) + " Number of steps: " + str(stepsTaken))  # Print episode information
        stepsPerEp[e] = stepsTaken  # Save the number of steps it took to complete the episode
    env.close()  # Close the simulator environment

    if makePlots:  # Generate plots
        plotResults(episodes, stepsPerEp, order, basis, alpha)
        plotQvals(agent, basis, order, alpha)
