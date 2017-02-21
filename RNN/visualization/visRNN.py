import matplotlib as mpl
import numpy as np
import matplotlib.cm as cmx
from matplotlib import colors
from classes.NetCell import NetCell

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

def prep_figure(data, fig, title='Network Visualization'):

    fig.subplots_adjust(left=0, bottom=0.02, right=1, top=0.98)
    fig.canvas.set_window_title(title)

    # Explicitly create the default background subplot
    ax = fig.add_subplot(111)

    # GRID =====================================================================================
    # ax.set_visible(True)
    # from matplotlib.ticker import MultipleLocator
    # spacing = 1  # This can be your user specified spacing.
    # minorLocator = MultipleLocator(spacing)
    # # Set minor tick locations.
    # ax.yaxis.set_minor_locator(minorLocator)
    # ax.xaxis.set_minor_locator(minorLocator)
    # # Set grid to use minor tick locations.
    # ax.grid(which='minor')
    # ax.grid(True, which='minor', c='green')
    # ==========================================================================================

    ax.set_aspect('equal')
    width = data.data_dim * 2 + data.hid_size + 6
    height = data.max_len * 2 + 2
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.tick_params(labelcolor=(1., 1., 1., 0.), top='off', bottom='off', left='off', right='off')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    return ax, width, height

def annotate(ax, height, data_dim, hid_size, base_font_size):
    text_header_y = height - 1
    inp_width = data_dim + 2
    hid_width = hid_size + 2

    labels = []
    labels.append(ax.text(3, text_header_y, r'$input$', ha='center', fontsize=base_font_size))
    labels.append(ax.text(inp_width + 2,  text_header_y, r'$hidden$', ha='center', fontsize=base_font_size))
    labels.append(ax.text(inp_width + hid_width + 2,  text_header_y, r'$output$', ha='center', fontsize=base_font_size))

    return labels

def deannotate(labels):
    for lbl in labels:
        lbl.remove()

def draw_epoch(ax, height, inp, hid, init, out, targ, colmap, nrange):

    data_dim = np.shape(inp)[1]
    hid_size = np.shape(hid)[1]
    inp_width = data_dim + 3
    hid_width = hid_size + 1

    row_grid_y = height - 5
    inp_grid_x = 2
    hid_grid_x = inp_width
    out_grid_x = inp_width + hid_width
    arrow_x = out_grid_x + 0.5
    targs = np.where(targ!=0)[1]

    # Drawing
    j=0
    cgrid(hid_grid_x, row_grid_y + 2, ax, np.reshape(init, (1, hid_size)), colmap, nrange)
    for i, h ,o, t in zip(inp, hid, out, targs):
        ax.text(inp_grid_x - 1, row_grid_y + 0.3, r'$t_{}$'.format(j), ha='center', fontsize=14)
        cgrid(inp_grid_x, row_grid_y, ax, np.reshape(i, (1, data_dim)), colmap, nrange)
        cgrid(hid_grid_x, row_grid_y, ax, np.reshape(h, (1, hid_size)), colmap, nrange)
        cgrid(out_grid_x, row_grid_y, ax, np.reshape(o, (1, data_dim)), colmap, nrange)
        ax.arrow(arrow_x + t, row_grid_y - 0.5, 0, 0, head_width=0.3, head_length=0.3, fc='k', ec='k', lw=2)
        ax.arrow(hid_grid_x - 1, row_grid_y + 0.5, 0.8, 0, head_width=0.1, head_length=0.2, fc='darkgray', ec='darkgray')
        ax.arrow(out_grid_x - 1, row_grid_y + 0.5, 0.8, 0, head_width=0.1, head_length=0.2, fc='darkgray', ec='darkgray')
        ax.arrow(hid_grid_x + (hid_size) / 2, row_grid_y + 2, 0, -0.8, head_width=0.1, head_length=0.2, fc='darkgray', ec='darkgray')
        j += 1
        row_grid_y -= 2

def main():
    from SRN.visualization.RNData import RNData
    import matplotlib.pyplot as plt
    data = RNData('/Users/alexten/Projects/PDP/SRN/logdir/Sess_2016-10-22_15-43-14/mpl_data/snaplog--7-31-3.pkl')

    data_dim = 7
    hid_size = 3

    fig = plt.figure()
    board = prep_figure(data, fig, 'Example vis')
    annotate(board, data_dim, hid_size, 12)

    # Pick a snap
    epoch = 10
    pattern_ind = 6
    max_len = 31
    snap = data.main[epoch]
    a = pattern_ind * max_len
    b = a + snap['seq_lens'][pattern_ind] - 1

    inp_sequence = snap['inp'][a:b]
    hid_sequence = snap['hid'][a:b]
    prevhid_sequence = np.vstack((np.zeros(hid_size), hid_sequence[0:-1]))
    out_sequence = snap['out'][a:b]
    targ_sequence = snap['targ'][a:b]

    draw_epoch(data, board, inp_sequence, hid_sequence, prevhid_sequence, out_sequence, targ_sequence, 'coolwarm', 1)

    plt.show()

if __name__=='__main__': main()