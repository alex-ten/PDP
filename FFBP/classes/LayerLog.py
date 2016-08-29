import numpy as np
import pickle

class LayerLog(object):
    def __init__(self, layer):
        self.name = layer.layer_name
        self.size = int(layer.size)
        self.inp = []
        self.W = []
        self.b = []
        self.netinp = []
        self.activations = []
        self.targ = []
        self.ded_W = []
        self.ded_b = []
        self.ded_netinp = []
        self.ded_activations = []

    def append(self, state):
        assert type(state) is zip, 'state parameter must be a zip object'
        for a, value in state:
            self.__getattribute__(a).append(value)
