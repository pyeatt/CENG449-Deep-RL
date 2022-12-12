import struct
import zmq

from agent_helpers import *


def main():
    epsilon = 1.0  # e greedy
    gamma = 0.95  # discount
    alpha = 0.85  # learning rate

    with open("./data.csv.npy", "rb") as file:
        q_values = np.load(file)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    while True:
        reward_old = 0
        not_crashed = True

        x, xdot, theta, thetadot = choose_random_demo_start()
        request_bytes = struct.pack('iffff', SET_STATE, x, xdot, theta, thetadot)
        socket.send(request_bytes)
        response_bytes = socket.recv()
        x, xdot, theta, thetadot, reward = struct.unpack('fffff', response_bytes[4:])
        x, xdot, theta, thetadot = bin_state(x, xdot, theta, thetadot)
        x_old, xdot_old, theta_old, thetadot_old, reward_old = x, xdot, theta, thetadot, reward
        action = get_action(x, xdot, theta, thetadot, epsilon, q_values)

        while not_crashed:
            x, xdot, theta, thetadot, reward = struct.unpack('fffff', response_bytes[4:])
            x, xdot, theta, thetadot = bin_state(x, xdot, theta, thetadot)
            if reward_old < 0:
                not_crashed = False
            old_q_val = q_values[action, x_old, xdot_old, theta_old, thetadot_old]
            td = reward_old + (gamma * np.max(q_values[:, x, xdot, theta, thetadot])) - old_q_val
            new_q_val = old_q_val + (alpha * td)
            q_values[action, x_old, xdot_old, theta_old, thetadot_old] = new_q_val
            action = get_action(x, xdot, theta, thetadot, epsilon, q_values)
            u = PUSH[action]
            x_old, xdot_old, theta_old, thetadot_old, reward_old = x, xdot, theta, thetadot, reward
            request_bytes = struct.pack('if', APPLY_FORCE, u)
            socket.send(request_bytes)
            response_bytes = socket.recv()

def train():
    epsilon = 0.9  # e greedy
    gamma = 0.95  # discount
    alpha = 0.85  # learning rate
    # q_values = np.zeros((ACTIONS, X_STATES, X_VELS, THETA_STATES, THETA_VELS), dtype=float)
    t_cycles = 100000000

    with open("./data.csv.npy", "rb") as file:
        q_values = np.load(file)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    while True:
        if t_cycles % 10 == 0:
            print(str(100000000-t_cycles))
            data = np.asarray(q_values)
            np.save('data.csv', data)
        count = 0
        reward_old = 0
        t_cycles -= 1
        not_crashed = True

        x, xdot, theta, thetadot = choose_random_start()
        request_bytes = struct.pack('iffff', SET_STATE, x, xdot, theta, thetadot)
        socket.send(request_bytes)
        response_bytes = socket.recv()
        x, xdot, theta, thetadot, reward = struct.unpack('fffff', response_bytes[4:])
        x, xdot, theta, thetadot = bin_state(x, xdot, theta, thetadot)
        x_old, xdot_old, theta_old, thetadot_old, reward_old = x, xdot, theta, thetadot, reward
        action = get_action(x, xdot, theta, thetadot, epsilon, q_values)

        while (count < 5000) and not_crashed:
            count += 1
            x, xdot, theta, thetadot, reward = struct.unpack('fffff', response_bytes[4:])
            # new_state = [x, xdot, theta, thetadot]
            x, xdot, theta, thetadot = bin_state(x, xdot, theta, thetadot)
            # bins = [x, xdot, theta, thetadot]
            # print("State:   " + str(new_state) + "    Reward:   " + str(reward) + "    Bins:    " + str(bins) + "    Cycle number:   " + str(100000000-t_cycles))
            if reward_old < 0:
                not_crashed = False

            old_q_val = q_values[action, x_old, xdot_old, theta_old, thetadot_old]
            td = reward_old + (gamma * np.max(q_values[:, x, xdot, theta, thetadot])) - old_q_val
            new_q_val = old_q_val + (alpha * td)
            q_values[action, x_old, xdot_old, theta_old, thetadot_old] = new_q_val
            action = get_action(x, xdot, theta, thetadot, epsilon, q_values)
            u = PUSH[action]
            x_old, xdot_old, theta_old, thetadot_old, reward_old = x, xdot, theta, thetadot, reward
            request_bytes = struct.pack('if', APPLY_FORCE, u)
            socket.send(request_bytes)
            response_bytes = socket.recv()

    print(q_values)
    data = np.asarray(q_values)
    np.save('data.csv', data)


if __name__ == "__main__":
    main()
    # train() # uncomment this and comment out the call to main to train the model
