# import sys
# sys.path.append('/Users/alexten/Projects/PDP/')

import time
import matplotlib.pyplot as plt
import FFBP.visualization.funcs as vf
from FFBP.visualization.NetworkData import NetworkData

plt.ion()

net_data = NetworkData('example.pkl')

logs = net_data.main.values()

fig, figure_map = vf.prep_figure(net_data, cpi = 4)

axs = vf.prep_all_axes(fig, figure_map, net_data)

vf.draw_all_layers(fig, logs, axs, 0, 0)

plt.show()
