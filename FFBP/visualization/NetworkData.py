import pickle
import numpy as np
from FFBP.visualization import visfuncs as vf

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
        rollup = vf.rollup(strip)
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