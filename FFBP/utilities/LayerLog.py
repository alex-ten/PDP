class LayerLog(object):
    def __init__(self, layer):
        self.name = layer.layer_name
        self.size = int(layer.size)
        self.sender_size = layer.sender_size
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

    def append(self, state):
        assert type(state) is zip, 'state parameter must be a zip object'
        for attr, value in state:
            if type(value) is list: value = value[0]
            self.__getattribute__(attr).append(value)
