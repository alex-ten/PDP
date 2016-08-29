import time
from FFBP.visualization.Artist import Artist
from FFBP.visualization.NetworkData import NetworkData

start=time.time()

xor = NetworkData('example_snap2.pkl')
xor.stdout()
xorplot = Artist(style_sheet='seaborn-dark')
xorplot.outline_all(xor)

xorplot.fill_axes(xor, 4500, c='jet', pattern=1, grid=False)
xorplot.remove_ticklabels()
print('Process time: {}'.format(time.time()-start))
xorplot.show()