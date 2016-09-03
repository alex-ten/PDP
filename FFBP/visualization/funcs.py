import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from matplotlib import colors
from IPython.display import display


import matplotlib

def v2c(x, colormap_string='coolwarm'):
    my_cmap = cm = plt.get_cmap(colormap_string)
    cNorm  = colors.Normalize(vmin=-1, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=my_cmap)
    return scalarMap.to_rgba(x)

def cgrid(xc, yc, ax, M):
    M = np.flipud(M)
    for (x, y), w in np.ndenumerate(M.T):
        color = v2c(w, 'coolwarm')
        rect = plt.Rectangle([x + xc,y + yc], 1, 1,
                             facecolor=color, edgecolor='#F7F7F7', linewidth=1)
        ax.add_patch(rect)

def prep_figure(data, cpi=5):
    # Set figure size based on number of layers and their sizes
    width = 9
    height = (data.num_units + 8 * data.num_layers) / cpi + (2 / cpi)
    ratio = height / width
    cell_size = 1 / (height * cpi)  # cell size as fraction of plot height and width == this would accommodate up to 50 x 50

    # Create the figure
    fig = plt.figure(num=None, figsize=(width, height), dpi=100)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.gcf().canvas.set_window_title('Network Visualization')

    # Explicitly create the default background subplot
    sbp = plt.subplot(111)
    sbp.set_visible(True)
    sbp.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
    origin = 0.05
    plt.ion()
    return fig, (origin, cell_size, ratio)

def prep_axes_pair(fig, origin, cell_size, ratio, layer):
    # ================== LOCAL VARIABLES ================== #
    num_attributes = 3
    sender_name, sender_size = layer.sender

    # Text coordinates:
    text_W_x= sender_size / 2 + 1
    text_b_x = sender_size + 2.5
    text_net_x = sender_size + 4.5
    text_a_x = sender_size + 6.5
    text_y = layer.size + 4.8
    xlim = sender_size + num_attributes * 2 + 2
    ylim = layer.size + 2 + 4

    # ================== AXES AND TEXT ================== #

    # FP (left) panel:
    w1 = cell_size * (sender_size + num_attributes * 2 + 2) * ratio
    h1 = cell_size * (layer.size + 2 + 4)
    x1 = 0.5 - w1 - 1 * cell_size * ratio
    y1 = origin

    ax1 = plt.axes([x1, y1, w1, h1])
    ax1.set_title(layer.name + ' FP', fontsize=10)
    ax1.set_xlim(0, xlim)
    ax1.set_ylim(0, ylim)

    ax1.text(text_W_x, text_y, r'$w$', ha='center', fontsize=11)
    ax1.text(text_b_x, text_y, r'$b$', ha='center', fontsize=11)
    ax1.text(text_net_x, text_y, r'$net$', ha='center', fontsize=11)
    ax1.text(text_a_x, text_y, r'$a$', ha='center', fontsize=11)
    ax1.text(1.1,1, r'${}$'.format(sender_name), ha='left', va='bottom', fontsize=10)

    ax1.spines['top'].set_visible(False); ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_color('#D4D4D4')
    ax1.spines['right'].set_color('#D4D4D4')
    ax1.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')

    x2 = 0.5 + 1 * cell_size * ratio
    y2 = origin
    w2 = w1
    h2 = h1

    # BP (left) panel:
    ax2 = plt.axes([x2, y2, w2, h2])
    ax2.set_title(layer.name + ' BP', fontsize=10)
    ax2.set_xlim(0, xlim)
    ax2.set_ylim(0, ylim)

    text_y += 0.1

    ax2.text(text_W_x, text_y, r'$\frac{\partial E}{\partial w}$', ha ='center', fontsize=13.5)
    ax2.text(text_b_x, text_y, r'$\frac{\partial E}{\partial b}$', ha ='center', fontsize=13.5)
    ax2.text(text_net_x, text_y, r'$\frac{\partial E}{\partial net}$', ha='center', fontsize=13.5)
    ax2.text(text_a_x, text_y, r'$\frac{\partial E}{\partial a}$', ha='center', fontsize=13.5)

    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_color('#D4D4D4')
    ax2.spines['left'].set_color('#D4D4D4')
    ax2.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')

    origin += h1 + cell_size * 2
    return origin, ax1, ax2

def prep_all_axes(fig, figure_map, data):
    origin, cell_size, ratio = figure_map
    axs = []
    for l in data.main.values():
        origin, ax1, ax2 = prep_axes_pair(fig, origin, cell_size, ratio, l)
        axs.append([ax1, ax2])
    return axs

def fill_axes_pair(fig, ax1, ax2, layer, epoch, pattern):
    sender_name, sender_size = layer.sender
    grid_b_x = sender_size + 2
    grid_net_x = sender_size + 4
    grid_a_x = sender_size + 6

    # FP (left) panel
    cgrid(1, 4, ax1, layer.W[epoch])
    cgrid(grid_b_x, 4, ax1, np.reshape(layer.b[epoch], (layer.size, 1)))
    cgrid(grid_net_x, 4, ax1, np.reshape(layer.net[epoch][pattern], (layer.size, 1)))
    cgrid(grid_a_x, 4, ax1, np.reshape(layer.act[epoch][pattern], (layer.size, 1)))
    cgrid(1, 2, ax1, np.reshape(layer.inp[epoch][pattern], (1, sender_size)))

    # BP (right) panel
    cgrid(1, 4, ax2, layer.ded_W[epoch])
    cgrid(grid_b_x, 4, ax2, np.reshape(layer.b[epoch], (layer.size, 1)))
    cgrid(grid_net_x, 4, ax2, np.reshape(layer.ded_net[epoch][pattern], (layer.size, 1)))
    cgrid(grid_a_x, 4, ax2, np.reshape(layer.ded_act[epoch][pattern], (layer.size, 1)))

def draw_all_layers(fig, logs, axes, epoch, pattern):
    for l, ax in zip(logs, axes):
        fill_axes_pair(fig, ax[0], ax[1], l, epoch, pattern)
    display(fig)
