from fourier_basis import FourierBasis
from polynomial_basis import PolynomialBasis
from mc import Car
import numpy as np
import random
import sys

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import animation
EPISODES = 1000
RUNS = 100

class SarsaLambda:
    def __init__(self, x, d, c, order=3, alpha=0.001, decay=.9, epsilon=.03, gamma=1):
        self.x = x
        # self.alpha = alpha
        self.decay = decay
        self.epsilon = epsilon
        self.actions = (-1, 0, 1)
        self.gamma = gamma
        self.d = d
        self.n = order # basis order
        self.w = np.zeros((order+1)**d, dtype='double')

        self.alpha = np.ones((order+1)**d)
        norm = np.linalg.norm(c, axis=1)
        norm[0] = 1
        self.alpha = alpha / norm

    def findValue(self, state, action):
        s_a_pair = (state[0], state[1], action)
        return np.dot(self.x(s_a_pair), self.w)


    def choose_action(self, state, epsilon): # .03
        if random.random() < epsilon:
            return random.choice(self.actions)

        best = [self.actions[0]]
        bestVal = self.findValue(state, self.actions[0])
        for action in self.actions[1:]:
            value = self.findValue(state, action)
            if value == bestVal:
                best.append(action)
            elif value > bestVal:
                bestVal = value
                best = [action]
        return random.choice(best)
    

    def run_episode(self, max_episode_length=3000):
        # initialize S
        car = Car()
        state = car.reset()

        # Choose A from policy
        A = self.choose_action(state, self.epsilon)

        # x = x (S, A)
        x = self.x([state[0], state[1], A])

        # \vec{z} = \vec{0}
        z = np.zeros(len(x)) # how long is the eligibility trace? 

        Qold = 0

        iteration = 0
        while True:
            iteration += 1
            R, stateP = car.apply_force(A)
            Ap = self.choose_action(stateP, self.epsilon)
            xp = self.x([stateP[0], stateP[1], Ap])
            Q = np.dot(self.w, x)
            Qp = np.dot(self.w, xp)
            delta = R + self.gamma * Qp - Q
            z = self.gamma * self.decay * z + (1 - self.alpha * self.gamma * self.decay * np.dot(z, x)) * x

            self.w = self.w + self.alpha * (delta  + Q - Qold)*z - self.alpha*(Q - Qold)*x
            Qold = Qp
            x = xp
            A = Ap
            if R == 0:
                return iteration

    def outputValueFunction(self, xspace, yspace):
        z = np.zeros((len(xspace), len(yspace)))
        for j, x in enumerate(xspace):
            for i, y in enumerate(yspace):
                bestAction = self.choose_action((x, y), 0)
                z[j, i] = -1 * self.findValue((x, y), bestAction)
        
        return z

    def initAnimation(self):
        self.animCar = Car()
        self.animCar.reset()

    def animationFunction(self, frameNumber):
        plt.clf()
        plt.title(str(frameNumber) + "/250")
        phase_shift = -.5 * 1#np.pi
        x = np.linspace(-1.2, .5, 100)
        y = -np.cos(x - phase_shift)
        plt.plot(x, y)
        state = (self.animCar.x, self.animCar.v)
        action=self.choose_action(state, 0)
        _, sp = self.animCar.apply_force(action)
        plt.plot(sp[0], -np.cos(sp[0] - phase_shift), 'ro')


if __name__ == "__main__":
    fig = plt.figure()
    dimensions = 3
    if "--animate" in sys.argv:
        if "--save-w" in sys.argv:
            print("WARNING: Not going to save weights")
        print("Animating (recommend having presaved weights that you can load for this...)")
        print("Only going to use order of 3 and Fourier basis")
        title = "results/result_fourier_n3_weights"
        basis = FourierBasis(3, 3)
        sl = SarsaLambda(basis.apply, c=basis.getC(), d=dimensions, order=3)
        if "--load-w" in sys.argv:
            print("loading weights")
            try:
                sl.w = np.loadtxt(title)
            except FileNotFoundError:
                print("Failed to load weights, make sure you have some saved")
                print("Specifically looking for " + title)
                exit(-1)
            except Exception as e:
                print("error")
                exit(-1)

        anim = animation.FuncAnimation(fig, sl.animationFunction, init_func=sl.initAnimation, interval=10, frames=250)
        plt.show()
        exit(0)

    orders = [3,5,7]
    print(f"There will be {RUNS} runs\nThere will be {EPISODES} episodes\nWill test orders {str(orders)}")

    x = np.linspace(-1.5, .5, 128)
    y = np.linspace(-.07, .07, 128)

    bases = [FourierBasis]
    # x, pi, alpha=.01, decay=.9, epsilon=.03):
    # sl = SarsaLambda(basis.apply, d=dimensions, order=orders[0])
    for i, b in enumerate(bases):
        for order in orders:
            # ex: result_fourier_n5
            title = f"results/result_{'fourier' if i==0 else 'poly'}"
            title+= f"_n{order}"
            basis = b(dimensions, order)
            all_iteration_data = []
            all_value_data = [] # will just contain the z values
            all_weight_data = []
            for run in range(RUNS):
                sl = SarsaLambda(basis.apply, c=basis.getC(), d=dimensions, order=order)
                if "--load-w" in sys.argv:
                    print("loading weights")
                    try:
                        sl.w = np.loadtxt(title+"_weights")
                    except FileNotFoundError:
                        print("Failed to load weights, make sure you have some saved")
                        exit(-1)
                    except Exception as e:
                        print("error")
                        exit(-1)
                print("Run: " + str(run))
                iteration_counts = []
                for k in range(EPISODES):
                    res = sl.run_episode()
                    iteration_counts.append(res)
                    if k % 200 == 0:
                        print("\tEpisode: ", k, res)

                all_iteration_data.append(iteration_counts)
                all_value_data.append(sl.outputValueFunction(x, y))
                all_weight_data.append(sl.w)

            # Average over run data
            average_per_episode = np.mean(all_iteration_data, axis=0)
            average_value_per_episode = np.mean(all_value_data, axis=0)
            average_weights_per_episode = np.mean(all_weight_data, axis=0)

            if "--save-w" in sys.argv:
                print("Saving average weights")
                np.savetxt(title + "_weights", average_weights_per_episode)

            
            if "--save-res" in sys.argv:
                print("Saving data (into results folder): " + title)
                np.savetxt(title + "_iterations", average_per_episode)
                np.savetxt(title + "_values", average_value_per_episode)



