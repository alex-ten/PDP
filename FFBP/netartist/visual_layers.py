import FFBP.netartist.VisClasses as vc
import time

start=time.time()

xor = vc.NetworkData('example_snap.pkl')
xorplot = vc.NetPlot(style_sheet='seaborn-dark')
xorplot.fields_for_data(xor)

xorplot.draw(xor, 0, c='coolwarm', pattern=3)
xorplot.make_ticklabels_invisible()
print('Process time: {}'.format(time.time()-start))
xorplot.show()