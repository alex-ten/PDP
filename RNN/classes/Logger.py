from PDPATH import PDPATH
import os

class Logger():
    def __init__(self, path=''):
        self.sess_index = 0
        self.parent_path = PDPATH() + '/RNN/logs/'
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
        dirname = 'RNNlog_{}'.format(self.sess_index)
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