from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np


def softmax(x):
    x = x - np.max(x, axis=1).reshape([-1,1])
    EXP = np.exp(x)
    return EXP / (np.sum(EXP, axis=1).reshape([-1,1]))


def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


def hist3d(M, xyz_labels=('X','Y','Z')):
    fignum = 1
    figlist = plt.get_fignums()
    if len(figlist):
        fignum = max(figlist) + 1
    fig = plt.figure(fignum, facecolor='w')
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel(xyz_labels[0])
    ax.set_ylabel(xyz_labels[2])
    ax.set_zlabel(xyz_labels[1])

    z_max = M.shape[1]
    z_colinds = list(range(0,z_max))
    z_coords = list(range(0,z_max*10,10))
    CMAP = get_cmap(z_max)
    for c, z, ys in zip(z_colinds, z_coords, M):
        # You can provide either a single color or an array. To demonstrate this,
        # the first bar of each set will be colored cyan.
        # cs = [c] * len(xs)
        # cs[0] = 'c'
        x_axis = list(range(len(ys)))
        ax.bar(x_axis, ys, zs=z, zdir='y', color=CMAP(c), alpha=0.7)
    return fig

def main():
    a = np.random.rand(5,10)

    print(get_cmap(10))

    f = hist3d(a, ['Category', 'Frequency', 'Time'])
    plt.show()

if __name__=="__main__": main()