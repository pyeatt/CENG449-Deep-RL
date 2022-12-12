import numpy as np
import matplotlib as plt
from matplotlib import pyplot as pl


def plotResults(episodes, stepsPerEp, order, basis, alpha):
    pl.plot(range(0, episodes), stepsPerEp)
    pl.title(basis + " bases order: " + str(order) + "\nAlpha = " + str(alpha))
    pl.xlabel("Episodes")
    pl.ylabel("Steps")
    pl.show()


def plotQvals(agent, basis, order, alpha):
    fig = plt.pyplot.figure()
    mesh = fig.add_subplot(projection='3d')
    x, y = np.meshgrid(np.linspace(0.0, 1.00, 200), np.linspace(0.0, 1.00, 200))
    z = np.zeros(x.shape)

    for i in range(0, z.shape[0]):
        for j in range(0, z.shape[1]):
            (qVal, _) = agent.maxQ([x[i, j], y[i, j]])
            z[i, j] = -1.0 * qVal

    surf = mesh.plot_surface(x, y, z, cmap=plt.cm.RdYlGn, linewidth=0, antialiased=False)
    mesh.view_init(elev=45, azim=45)
    pl.title('Cost-to-go Function for ' + basis + ' basis of order ' + str(order) + "\nAlpha = " + str(alpha))
    fig.colorbar(surf, shrink=0.8, aspect=5)
    pl.show()
