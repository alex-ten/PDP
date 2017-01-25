import os

class LayerLog(object):
    def __init__(self, layer):
        self.name = layer.layer_name
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
    def __init__(self):
        self.sess_index = 0
        self.parent_path = os.getcwd() + '/FFBP/logs'
        self.may_be_make_parent()
        self.child_path = self.may_be_make_child()


    def may_be_make_parent(self):
        try:
            os.mkdir(self.parent_path)
        except FileExistsError:
            pass

    def may_be_make_child(self):
        last_ind = self.get_last()
        if last_ind is not None:
            self.sess_index = last_ind + 1
        dirname = 'FFBPlog_{}'.format(self.sess_index)
        child_path = self.parent_path + '/' + dirname
        os.mkdir(child_path)
        return child_path

    def get_last(self):
        contents = os.listdir(self.parent_path)
        inds = [int(os.path.splitext(x)[0].split(sep='/')[-1].split(sep='_')[-1]) for x in contents]
        try:
            return sorted(inds).pop()
        except IndexError:
            return None
