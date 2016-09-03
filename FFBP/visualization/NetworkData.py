import pickle
import numpy as np

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
        self.layer_names = self.main.keys()
        self.num_layers = len(self.main)
        self.num_units = sum([x.size for x in self.main.values()])

    def stdout(self):
        np.set_printoptions(precision=3, suppress=True, linewidth=200)
        print('Error:')
        print(self.error, end='\n\n')
        for layer, log in self.main.items():
            print('>>> ' + layer + ':')
            for k, v in log.__dict__.items():
                if k=='name' or k=='size' or k=='targ' or k=='sender_size': continue
                a = len(k)
                print('    ' + '{}:'.format(k) + ' '*(20-a) + '{} snaps of {} array'.format(len(v), np.shape(v[0])))
            print('===' * 50, end='\n\n')