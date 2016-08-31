import time
from FFBP.visualization.Artist import Artist
import numpy as np
from FFBP.visualization.NetworkData import NetworkData

start=time.time()

xor = NetworkData('snap_8t.pkl')
xor.stdout()
# xorplot = Artist(style_sheet='seaborn-dark')
# xorplot.outline_all(xor)
#
# xorplot.fill_axes(xor, 4500, c='jet', pattern=1, grid=False)
# xorplot.remove_ticklabels()
# print('Process time: {}'.format(time.time()-start))
# xorplot.show()