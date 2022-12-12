"""
This is a function to render a postion list to an animation.

It is modified a little to make it work with the generated position list. 

Here is the github file it is from:
https://github.com/mpatacchiola/dissecting-reinforcement-learning/blob/master/src/6/mountain-car/mountain_car.py
"""


from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np

def render(filename, position_list):
    delta_t = 0.1
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.2, 0.5), ylim=(-1.1, 1.1))
    ax.grid(False)  # disable the grid
    ax.set_title(filename)
    x_sin = np.linspace(start=-1.2, stop=0.5, num=100)
    y_sin = np.sin(3 * x_sin)
    ax.plot(x_sin, y_sin)  # plot the sine wave
    dot, = ax.plot([], [], 'ro')
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    _position_list = position_list
    _delta_t = delta_t

    def _init():
        dot.set_data([], [])
        time_text.set_text('')
        return dot, time_text

    def _animate(i):
        x = _position_list[i]
        y = np.sin(3 * x)
        dot.set_data(x, y)
        time_text.set_text("Time: " + str(np.round(i*_delta_t, 1)) + "s" + '\n' + "Frame: " + str(i))
        return dot, time_text

    ani = animation.FuncAnimation(fig, _animate, np.arange(1, len(position_list)),
                                    blit=True, init_func=_init, repeat=False)

    filename = filename + ".gif"
    ani.save(filename, writer='imagemagick', fps=int(1/delta_t))

    fig.clear()
    plt.close(fig)