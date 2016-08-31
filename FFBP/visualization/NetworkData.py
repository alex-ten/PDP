import pickle
import numpy as np
from FFBP.visualization import visfuncs as vf

class NetworkData(object):
    def __init__(self, path):
        with open(path, 'rb') as opened_file:
            snapshot = pickle.load(opened_file)
        self.error = snapshot['error']
        self.checkpoints = snapshot['checkpoints']
        self.attributes = snapshot['attributes']
        del snapshot['error']
        del snapshot['checkpoints']
        del snapshot['attributes']
        self.main = snapshot
        self.network_size = len(self.main)
        self.layer_names = self.main.keys()

    def stdout(self):
        np.set_printoptions(precision=3, suppress=True, linewidth=200)
        print('Error:')
        print(self.error, end='\n\n')
        for layer, log in self.main.items():
            print('>>> ' + layer + ':')
            for k, v in log.__dict__.items():
                if k=='name' or k=='size' or k=='targ': continue
                print('    ' + k + ':')
                print('        '+'{} snaps of {} array'.format(len(v), np.shape(v[0])))
            print('===' * 50, end='\n\n')