import numpy as np
import pickle

# store Snapshots in a dict, keyed by layer names. Then loop over each snapshot to append them with new state

class Snapshot(object):
    def __init__(self, layer, checkpoint):
        self.size = int(layer.size)
        self.inp = np.empty()
        self.W = np.empty()
        self.b = np.empty()
        self.neting = np.empty()
        self.act = np.empty()
        self.targ = np.empty()
        self.ded_W = np.empty()
        self.ded_b = np.empty()
        self.ded_neting = np.empty()
        self.ded_act = np.empty()

    def append(self, state):
        pass

# pickle thyself:

    # def Load(self):
    #     f = open(self.filename, 'rb')
    #     tmp_dict = pickle.load(f)
    #     f.close()
    #
    #     self.__dict__.update(tmp_dict)
    #
    # def Save(self):
    #     f = open(self.filename, 'wb')
    #     cPickle.dump(self.__dict__, f, 2)
    #     f.close()