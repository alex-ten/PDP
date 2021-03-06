import matplotlib as mpl
import numpy as np
import matplotlib.cm as cmx
from matplotlib import colors
from matplotlib.patches import Rectangle

class NetCell(Rectangle):
    def __init__(self, *args, **kwargs):
        self.cellval = kwargs.pop('cellval')
        self.inds = kwargs.pop('inds')
        super(NetCell, self).__init__(*args, **kwargs)
    def get_cellval(self):
        return self.cellval
    def get_inds(self):
        return self.inds

def v2c(x, colmap, nrange):
    my_cmap = cm = mpl.cm.get_cmap(colmap)
    cNorm  = colors.Normalize(vmin=-nrange, vmax=nrange)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=my_cmap)
    return scalarMap.to_rgba(x)

def cgrid(xc, yc, ax, M, colmap, nrange, t=False):
    nrows = np.shape(M)[0]
    for (x, y), w in np.ndenumerate(np.flipud(M).T):
        color = v2c(w, colmap, nrange)
        rect = NetCell([x + xc,y + yc], 1, 1,
                        facecolor=color,
                        edgecolor='#F7F7F7',
                        linewidth=1,
                        cellval=w,
                        inds=(nrows - y,x+1),
                        picker=True)
        if t:
            rect.set_edgecolor('black')
        ax.add_patch(rect)

def prep_figure(data, fig):
    fig.subplots_adjust(left=0, bottom=0.02, right=1, top=0.98)
    fig.canvas.set_window_title('Network Visualization')

    # Explicitly create the default background subplot
    panel_board = fig.add_subplot(111)

    # GRID =====================================================================================
    # panel_board.set_visible(True)
    # from matplotlib.ticker import MultipleLocator
    # spacing = 1  # This can be your user specified spacing.
    # minorLocator = MultipleLocator(spacing)
    # # Set minor tick locations.
    # panel_board.yaxis.set_minor_locator(minorLocator)
    # panel_board.xaxis.set_minor_locator(minorLocator)
    # # Set grid to use minor tick locations.
    # panel_board.grid(which='minor')
    # panel_board.grid(True, which='minor', c='green')
    # GRID =====================================================================================

    panel_board.set_aspect('equal')
    width = max([l.sender[1] for l in data.main.values()])
    height = data.num_units
    panel_board.set_xlim(0, (width + 6) * 2 + 1) # set width + x-padding for figure here
    panel_board.set_ylim(0, height + (6 * data.num_layers)) # set padding for each panel here
    panel_board.tick_params(labelcolor=(1., 1., 1., 0.), top='off', bottom='off', left='off', right='off')
    panel_board.spines['top'].set_visible(False)
    panel_board.spines['bottom'].set_visible(False)

    return panel_board

def annotate_layer(origin, ax, layer, base_font_size):
    targ = False
    num_attributes = 3
    sender_name, sender_size = layer.sender

    # Some special treatment for a layer with a target vector
    if len(layer.t):
        targ = True
        num_attributes += 1

    labels = []

    # Grid panel
    # Placement
    subpanel_height = layer.size + 2 + 4

    text_W_x = sender_size / 2 + 1
    text_b_x = sender_size + 2.5
    text_net_x = sender_size + 4.5
    text_a_x = sender_size + 6.5
    text_y = origin + layer.size + 4.5
    text_inp_y = origin + 1.5
    text_dednet_x = sender_size + 11.5
    text_deda_x = sender_size + 13.5

    title_x = text_W_x
    title_y = origin + subpanel_height - 0.5

    labels.append(ax.text(title_x, title_y, r'${}$'.format(layer.name), ha='center', va='center', fontsize=base_font_size+base_font_size*0.5))

    labels.append(ax.text(text_W_x, text_y, r'$w$', ha='center', fontsize=base_font_size))
    labels.append(ax.text(text_b_x, text_y, r'$b$', ha='center', fontsize=base_font_size))
    labels.append(ax.text(text_net_x, text_y, r'$net$', ha='center', fontsize=base_font_size))
    labels.append(ax.text(text_a_x, text_y, r'$a$', ha='center', fontsize=base_font_size))
    labels.append(ax.text(1, text_inp_y, r'${}$'.format(sender_name), ha='left', va='top', fontsize=base_font_size-base_font_size*0.1))
    if targ:
        labels.append(ax.text(sender_size + 8.5, text_y, r'$t$', ha='center', fontsize=base_font_size))
    labels.append(ax.text(text_dednet_x, text_y, r'$\frac{\partial E}{\partial net}$', ha='center', fontsize=base_font_size+base_font_size*0.2))
    labels.append(ax.text(text_deda_x, text_y, r'$\frac{\partial E}{\partial a}$', ha='center', fontsize=base_font_size+base_font_size*0.2))

    origin += subpanel_height

    return origin, labels

def annotate(snap, panel_board, fs):
    origin = 0
    label_groups = []
    for l in snap.main.values():
        origin, labels = annotate_layer(origin, panel_board, l, fs)
        label_groups.append(labels)

    return label_groups

def labels_off(label_groups):
    for group in label_groups:
        for label in group:
            label.remove()

def layer_image(origin, ax, layer, epoch, pattern, colmap, nrange):
    targ = False
    num_attributes = 3
    sender_name, sender_size = layer.sender

    # Some special treatment for a layer with a target vector
    if len(layer.t):
        targ = True
        num_attributes += 1

    subpanel_height = layer.size + 2 + 4

    # Grid panel
    # Placement
    grid_W_x = 1
    grid_b_x = sender_size + 2
    grid_net_x = sender_size + 4
    grid_a_x = sender_size + 6
    grid_l_y = origin + 4
    grid_inp_y = origin + 2
    grid_dednet_x = sender_size + 11
    grid_deda_x = sender_size + 13

    # Drawing
    cgrid(grid_W_x, grid_l_y, ax, layer.W[epoch].T, colmap, nrange)
    cgrid(grid_b_x, grid_l_y, ax, np.reshape(layer.b[epoch], (layer.size, 1)), colmap, nrange)
    cgrid(grid_net_x, grid_l_y, ax, np.reshape(layer.net[epoch][pattern], (layer.size, 1)), colmap, nrange)
    cgrid(grid_a_x, grid_l_y, ax, np.reshape(layer.act[epoch][pattern], (layer.size, 1)), colmap, nrange)
    cgrid(grid_W_x, grid_inp_y, ax, np.reshape(layer.inp[epoch][pattern], (1, sender_size)), colmap, nrange)
    # More special treatment for a layer with a target vector
    if targ:
        cgrid(sender_size + 8, grid_l_y, ax, np.reshape(layer.t[epoch][pattern], (layer.size, 1)), colmap, nrange, t=True)

    cgrid(grid_dednet_x, grid_l_y, ax, np.reshape(layer.ded_net[epoch][pattern], (layer.size, 1)), colmap, nrange)
    cgrid(grid_deda_x, grid_l_y, ax, np.reshape(layer.ded_act[epoch][pattern], (layer.size, 1)), colmap, nrange)

    origin += subpanel_height

    return origin

def draw_all_layers(snap, ax, epoch, pattern, colmap='coolwarm', nrange=4):
    origin = 0
    for l in snap.main.values():
        origin = layer_image(origin, ax, l, epoch, pattern, colmap, nrange)