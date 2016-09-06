import matplotlib as mpl
import matplotlib.pyplot as plt
if mpl.get_backend() != 'Qt4Agg': plt.switch_backend('qt4agg')

import numpy as np
import matplotlib.cm as cmx
from matplotlib import colors

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

def prep_figure(data, fig): # My display is 2560 x 1600 pixels

    fig.subplots_adjust(left=0, bottom=0.02, right=1, top=0.98)
    plt.gcf().canvas.set_window_title('Network Visualization')

    # Explicitly create the default background subplot

    sidebar = plt.subplot2grid((1,10), (0,0), colspan=1)
    sidebar.plot([1, 2, 3], lw=3, c='purple')
    sidebar.plot([3, 2, 1], lw=3, c='orange')
    sidebar.tick_params(labelcolor=(1., 1., 1., 0.), top='off', bottom='off', left='off', right='off')

    panels = plt.subplot2grid((1, 11), (0, 1), colspan=10)

    # todo =====================================================================================
    # panels.set_visible(True)
    # from matplotlib.ticker import MultipleLocator
    # spacing = 1  # This can be your user specified spacing.
    # minorLocator = MultipleLocator(spacing)
    # # Set minor tick locations.
    # panels.yaxis.set_minor_locator(minorLocator)
    # panels.xaxis.set_minor_locator(minorLocator)
    # # Set grid to use minor tick locations.
    # panels.grid(which='minor')
    # panels.grid(True, which='minor', c='green')
    # todo =====================================================================================

    panels.set_aspect('equal')
    width = max([l.sender[1] for l in data.main.values()])
    height = data.num_units
    panels.set_xlim(0, (width + 8) * 2) # set padding for each panel here
    panels.set_ylim(0, height + (6 * data.num_layers)) # set padding for each panel here
    panels.tick_params(labelcolor=(1., 1., 1., 0.), top='off', bottom='off', left='off', right='off')
    panels.spines['top'].set_visible(False)
    panels.spines['bottom'].set_visible(False)

    return panels

def layer_labels(origin, ax, layer):
    num_attributes = 3
    sender_name, sender_size = layer.sender

    # FP (left) panel:
    subpanel_width = sender_size + num_attributes * 2 + 2
    subpanel_height = layer.size + 2 + 4

    text_W_x = sender_size / 2 + 1
    text_b_x = sender_size + 2.5
    text_net_x = sender_size + 4.5
    text_a_x = sender_size + 6.5
    text_y = origin + layer.size + 4.5
    text_inp_y = origin + 1.5

    title_x = subpanel_width
    title_y = origin + subpanel_height - 0.5

    ax.text(title_x, title_y, r'${}$'.format(layer.name), ha='center', va='center', fontsize=12)

    ax.text(text_W_x, text_y, r'$w$', ha='center', fontsize=11)
    ax.text(text_b_x, text_y, r'$b$', ha='center', fontsize=11)
    ax.text(text_net_x, text_y, r'$net$', ha='center', fontsize=11)
    ax.text(text_a_x, text_y, r'$a$', ha='center', fontsize=11)
    ax.text(1, text_inp_y, r'${}$'.format(sender_name), ha='left', va='top', fontsize=10)

    # BP (right) panel:
    text_W_x += subpanel_width
    text_b_x += subpanel_width
    text_net_x += subpanel_width
    text_a_x += subpanel_width

    ax.text(text_W_x, text_y, r'$\frac{\partial E}{\partial w}$', ha='center', fontsize=13.5)
    ax.text(text_b_x, text_y, r'$\frac{\partial E}{\partial b}$', ha='center', fontsize=13.5)
    ax.text(text_net_x, text_y, r'$\frac{\partial E}{\partial net}$', ha='center', fontsize=13.5)
    ax.text(text_a_x, text_y, r'$\frac{\partial E}{\partial a}$', ha='center', fontsize=13.5)

    origin += subpanel_height

    return origin

def layer_image(origin, fig, ax, layer, epoch, pattern):
    sender_name, sender_size = layer.sender
    num_attributes = 3

    # FP (left) panel
    # Placement
    grid_W_x = 1
    grid_b_x = sender_size + 2
    grid_net_x = sender_size + 4
    grid_a_x = sender_size + 6
    grid_l_y = origin + 4
    grid_inp_y = origin + 2

    # Drawing
    cgrid(grid_W_x, grid_l_y, ax, layer.W[epoch])
    cgrid(grid_b_x, grid_l_y, ax, np.reshape(layer.b[epoch], (layer.size, 1)))
    cgrid(grid_net_x, grid_l_y, ax, np.reshape(layer.net[epoch][pattern], (layer.size, 1)))
    cgrid(grid_a_x, grid_l_y, ax, np.reshape(layer.act[epoch][pattern], (layer.size, 1)))
    cgrid(grid_W_x, grid_inp_y, ax, np.reshape(layer.inp[epoch][pattern], (1, sender_size)))

    subpanel_width = sender_size + num_attributes * 2 + 2
    subpanel_height = layer.size + 2 + 4

    # BP (right) panel
    # Placement
    grid_W_x += subpanel_width
    grid_b_x += subpanel_width
    grid_net_x += subpanel_width
    grid_a_x += subpanel_width

    # Drawing
    cgrid(grid_W_x, grid_l_y, ax, layer.ded_W[epoch])
    cgrid(grid_b_x, grid_l_y, ax, np.reshape(layer.b[epoch], (layer.size, 1)))
    cgrid(grid_net_x, grid_l_y, ax, np.reshape(layer.ded_net[epoch][pattern], (layer.size, 1)))
    cgrid(grid_a_x, grid_l_y, ax, np.reshape(layer.ded_act[epoch][pattern], (layer.size, 1)))

    origin += subpanel_height

    fig.canvas.draw()
    plt.pause(0.001)

    return origin

def draw_all_layers(snap, fig, ax, epoch, pattern):
    origin = 0
    for l in snap.main.values():
        origin = layer_image(origin, fig, ax, l, epoch, pattern)

def main(height = 1280, width = 700, screen_dpi = 96):
    from FFBP.visualization.NetworkData import NetworkData

    # Data
    snap = NetworkData('visualization/example1.pkl')

    # Figure
    fig = plt.figure(2, figsize=(height / screen_dpi, width / screen_dpi), facecolor='w', dpi=screen_dpi)
    panels = prep_figure(snap, fig)

    # Annotate
    origin = 0
    for l in snap.main.values():
        origin = layer_labels(origin, panels, l)

    draw_all_layers(snap, fig, panels, 0, 0)
    input('hit enter to update')

    draw_all_layers(snap, fig, panels, 8, 0)
    input('hit enter to quit')

if __name__=='__main__':
    main()