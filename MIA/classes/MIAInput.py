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
    def __init__(self, word, batch_size):
        features_path = os.getcwd() + '/MIA/raw/'
        with open(features_path + 'features.pkl', 'rb') as f:
            features = pickle.load(f)
        x0,x1,x2 = new_input(word, features, batch_size)
        self.word = word
        self.l0 = x0
        self.l1 = x1
        self.l2 = x2
        self.batch_size = batch_size

    def __call__(self):
        return (self.l0, self.l1, self.l2)