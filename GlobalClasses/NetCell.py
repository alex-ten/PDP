from matplotlib.patches import Rectangle

class NetCell(Rectangle):
    def __init__(self, *args, **kwargs):
        self.cellval = kwargs.pop('cellval')
        self.inds = kwargs.pop('inds')
        super(NetCell, self).__init__(*args, **kwargs)
    def get_cellval(self):
        return self.cellval
    def get_inds(self):
        return self.inds