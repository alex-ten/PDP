import pickle
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import FFBP.netartist.visfuncs as myfuncs
from matplotlib import style


class NetworkData(object):
    def __init__(self, path):
        with open(path, 'rb') as opened_file:
            snapshot = pickle.load(opened_file)
        self.error = snapshot['error']
        del snapshot['error']
        self.main = snapshot
        self.size = len(self.main)
        self.lnames = self.main.keys()
        self.checkpoints = self.error[:,0].astype(int)

    def get(self, l, var, epoch):
        row_ind = int(np.where(self.checkpoints == epoch)[0][0])
        strip = self.main[l][var][row_ind, :]
        rollup = myfuncs.rollup(strip)
        return rollup

    def stdout(self):
        np.set_printoptions(precision=3, suppress=True, linewidth=200)
        print('Error:')
        print(self.error, end='\n\n')
        for keys, subdicts in self.main.items():
            print('>>> ' + keys + ':')
            for k, v in subdicts.items():
                print('    ' + k + ':')
                print(v)
            print('===' * 50, end='\n\n')

class NetPlot(object):
    def __init__(self, fig_num = 1, fig_dims = (16,9), style_sheet = 'fivethirtyeight'):
        self.fig_style = style.use(style_sheet)
        # Plot fields
        self.omni = plt.figure(fig_num, fig_dims, facecolor='#F7F7F7')
        self.omni_title = plt.gcf().canvas.set_window_title('Network Visualization')
        self.meta = gridspec.GridSpec(2,3, width_ratios=[1,20,1],height_ratios=[20,1])
        self.hyper = None
        self.super = collections.OrderedDict()
        self.sub = collections.OrderedDict()

    def hyper_into_super(self, network):
        # Divide hyperfield by superfields
        assert network.size <= 16, 'Grid object cannot have more than 16 cells'
        if network.size == 1:
            a, b = 1, 1
        elif network.size == 2:
            a, b = 2, 1
        elif network.size == 3:
            a, b = 3, 1
        elif network.size == 4:
            a, b = 2, 2
        elif network.size > 4 and network.size <= 6:
            a, b = 2, 3
        elif network.size > 6 and network.size <= 9:
            a, b = 3, 3
        elif network.size > 9 and network.size <= 12:
            a, b = 3, 4
        else:
            a, b = 4, 4
        self.hyper = gridspec.GridSpecFromSubplotSpec(a, b, subplot_spec=self.meta[1])

    def super_into_basic(self, network):
        # divide each superfield by 2 basic fields
        for ind, lname in enumerate(network.lnames):
            field = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=self.hyper[ind])
            self.super[lname] = field
            tax = plt.subplot(field[0:])
            tax.set_title(lname, fontsize=15, position=(0.5, 1.05))
            tax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
            tax._frameon = False

    def prepare_basic(self, network):
        for layer in network.lnames:
            self.sub[layer] = {}
            self.sub[layer]['fp'] = myfuncs.fp_axes(self.super[layer][0], self.omni)
            self.sub[layer]['bp'] = myfuncs.bp_axes(self.super[layer][1], self.omni)

    def fields_for_data(self, network):
        self.hyper_into_super(network)
        self.super_into_basic(network)
        self.prepare_basic(network)

    def draw(self, network, epoch, pattern=0, c ='coolwarm'):
        pattern = int(pattern)
        for layer, subfields in self.sub.items():
            for subfield, vals in subfields.items():
                for key, ax in vals.items():
                    pre = network.get(layer, key, epoch)
                    if key == 'W' or key == 'ded_W':
                        img = pre
                    elif key=='b' or key=='ded_b':
                        img = pre.T
                    elif key == 'inp':
                        img = np.reshape(pre[pattern], (1, np.shape(pre)[1]))
                    else:
                        img = np.reshape(pre[pattern], (1, np.shape(pre)[1])).T
                    rows, cols = np.shape(img)
                    cax = ax.matshow(img, extent=[0,cols,0,rows], interpolation='nearest', cmap=c, vmin=-1, vmax=1)
                    ax.set(adjustable='box-forced', aspect='equal')
                    ax.set_axis_bgcolor('#F7F7F7')
                    ax.grid(True, linewidth=3, color='#F7F7F7')

    def make_ticklabels_invisible(self, mark_axes=False):
        for i, ax in enumerate(self.omni.axes):
            if mark_axes: ax.text(0.5, 0.5, "ax%d" % (i + 1), va="center", ha="center")
            for tl in ax.get_xticklabels() + ax.get_yticklabels():
                tl.set_visible(False)

    def show(self, tight=False):
        if tight: plt.tight_layout(pad=1)
        plt.show()

    def close(self):
        plt.close()