import numpy as np
import pickle
import os


def new_input(s, features, batch_size = 1):
    # special symbols
    alphabet = list('abcdefghijklmnopqrstuvwxyz_#@?')
    xs = list(s.lower())
    inds = [alphabet.index(x) for x in xs]
    return [np.tile(features[ind], [batch_size, 1]).T for ind in inds]


class MIAInput(object):
    def __init__(self, s, batch_size):
        features_path = os.getcwd() + '/MIA/raw/'
        with open(features_path + 'features.pkl', 'rb') as f:
            features = pickle.load(f)
        x0,x1,x2 = new_input(s, features, batch_size)
        self.word = s
        self.l0 = x0
        self.l1 = x1
        self.l2 = x2
        self.batch_size = batch_size

    def __call__(self):
        return (self.l0, self.l1, self.l2)

    def mask(self, l_pos, features):
        ll = [self.l0, self.l1, self.l2]
        fa0 = np.array(features)*2
        fa1 = fa0 + 1
        ll[l_pos][fa0] = 0
        ll[l_pos][fa1] = 0