import time
from FFBP.netartist.Artist import Artist
from FFBP.netartist.NetworkData import NetworkData

start=time.time()

xor = NetworkData('example_snap.pkl')
xorplot = Artist(style_sheet='seaborn-dark')
xorplot.outline_all(xor)

xorplot.fill_axes(xor, 0, c='coolwarm', pattern=3)
xorplot.remove_ticklabels()
print('Process time: {}'.format(time.time()-start))
xorplot.show()