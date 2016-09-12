import datetime as dt
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


def logdir():
    # Create logdir directory if it doesn't exist
    logdir_dir = os.getcwd() + '/logdir'
    try:
        os.mkdir(logdir_dir)
        # Name sess directory according to current date-time
        sess_name = 'Sess_' + dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        dir_path = logdir_dir + '/' + sess_name
        os.mkdir(dir_path)
        os.mkdir(dir_path + '/tf_params')
        os.mkdir(dir_path + '/mpl_data')
    except FileExistsError:
        sess_name = 'Sess_'+dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        dir_path = logdir_dir + '/' + sess_name
        os.mkdir(dir_path)
        os.mkdir(dir_path + '/tf_params')
        os.mkdir(dir_path + '/mpl_data')
    return dir_path

