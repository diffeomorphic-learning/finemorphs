import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors


def plotClassifRegress(positions1, colors1, positions2, colors2, positions3=None, colors3=None, fig=None,
                       title1='', title2='', title3='', vmin_vmax_flag=False):
    fig.clf()
    delta = 1e-6
    axes1 = [positions1[:, 0].min(), positions1[:, 0].max(), positions1[:, 1].min(), positions1[:, 1].max(),
             positions1[:, 2].min(), positions1[:, 2].max()]
    axes2 = [positions2[:, 0].min(), positions2[:, 0].max(), positions2[:, 1].min(), positions2[:, 1].max(),
             positions2[:, 2].min(), positions2[:, 2].max()]
    if positions3 is not None:
        axes3 = [positions3[:, 0].min(), positions3[:, 0].max(), positions3[:, 1].min(), positions3[:, 1].max(),
                 positions3[:, 2].min(), positions3[:, 2].max()]
    cmap = plt.cm.jet
    if vmin_vmax_flag:
        vmin = np.amin(np.ravel(colors1))
        vmax = np.amax(np.ravel(colors1))
    else:
        vmin1 = np.amin(np.ravel(colors1))
        vmax1 = np.amax(np.ravel(colors1))
        vmin2 = np.amin(np.ravel(colors2))
        vmax2 = np.amax(np.ravel(colors2))
        if positions3 is not None:
            vmin3 = np.amin(np.ravel(colors3))
            vmax3 = np.amax(np.ravel(colors3))
            vmin = np.amin(np.array([vmin1, vmin2, vmin3]))
            vmax = np.amax(np.array([vmax1, vmax2, vmax3]))
        else:
            vmin = np.amin(np.array([vmin1, vmin2]))
            vmax = np.amax(np.array([vmax1, vmax2]))
    norm = matplotlib.colors.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    c1 = np.ravel(colors1)
    c2 = np.ravel(colors2)
    if positions3 is not None:
        c3 = np.ravel(colors3)

    if positions3 is not None:
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')
    else:
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
    ax1.set_xlim3d(axes1[0], axes1[1])
    ax1.set_ylim3d(axes1[2], axes1[3])
    ax1.set_zlim3d(axes1[4], axes1[5])
    ax1.scatter(positions1[:, 0], positions1[:, 1], positions1[:, 2], marker='o', s=2, c=cmap(norm(c1)))
    ax1.set_title(title1)
    ax2.set_xlim3d(axes2[0], axes2[1])
    ax2.set_ylim3d(axes2[2], axes2[3])
    ax2.set_zlim3d(axes2[4], axes2[5])
    ax2.scatter(positions2[:, 0], positions2[:, 1], positions2[:, 2], marker='o', s=2, c=cmap(norm(c2)))
    ax2.set_title(title2)
    if positions3 is not None:
        ax3.set_xlim3d(axes3[0], axes3[1])
        ax3.set_ylim3d(axes3[2], axes3[3])
        ax3.set_zlim3d(axes3[4]-delta, axes3[5]+delta)
        ax3.scatter(positions3[:, 0], positions3[:, 1], positions3[:, 2], marker='o', s=2, c=cmap(norm(c3)))
        ax3.set_title(title3)
    plt.tight_layout()

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    time.sleep(.5)
    return fig