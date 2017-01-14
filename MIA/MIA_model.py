import code

from MIA.classes.MIANetwork import MIANetwork

OLgivenW = 10
OFgivenL = 10

mynet = MIANetwork(OLgivenW, OFgivenL, name='MIA')
mynet.run_sim('#ex')


code.interact(local = locals())