import os
from PDPATH import PDPATH

class LayerLog(object):
    def __init__(self, layer):
        self.name = layer.name
        self.size = int(layer.size)
        self.sender = (layer.sender_name,layer.sender_size)
        self.inp = []
        self.W = []
        self.b = []
        self.net = []
        self.act = []
        self.targ = []
        self.ded_W = []
        self.ded_b = []
        self.ded_net = []
        self.ded_act = []
        self.t = []

    def append(self, state):
        assert type(state) is zip, 'state parameter must be a zip object'
        for attr, value in state:
            if type(value) is list: value = value[0]
            self.__getattribute__(attr).append(value)


class Logger():
    def __init__(self, path=''):
        self.sess_index = 0
        self.parent_path = PDPATH() + '/FFBP{}/logs'.format('/'+ path if len(path) else '')
        self.may_be_make_parent()
        self.child_path = self.make_child_i(self.parent_path)

    def may_be_make_parent(self):
        try:
            os.mkdir(self.parent_path)
        except FileExistsError:
            pass

    def make_child_i(self, parent):
        last_ind = self.get_last(parent, 'FFBPlog')
        if last_ind is not None:
            self.sess_index = last_ind + 1
        dirname = 'FFBPlog_{}'.format(self.sess_index)
        child_path = parent + '/' + dirname
        os.mkdir(child_path)
        self.logs_child_path = child_path
        return child_path

    def get_last(self, parent, name):
        contents = os.listdir(parent)
        contents = [x for x in contents if x.startswith(name)]
        inds = [int(os.path.splitext(x)[0].split(sep='/')[-1].split(sep='_')[-1]) for x in contents]
        try:
            return sorted(inds).pop()
        except IndexError:
            return None