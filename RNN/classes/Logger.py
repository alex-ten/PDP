from PDPATH import PDPATH
import os

class Logger():
    def __init__(self, path=''):
        self.sess_index = 0
        self.logs_path = PDPATH() + '/RNN/logs/'
        self.trained_path = PDPATH() + '/RNN/trained_models/'
        self.may_be_make_dir(self.logs_path)
        self.may_be_make_dir(self.trained_path)

    def may_be_make_dir(self, dir):
        try:
            os.mkdir(dir)
        except FileExistsError:
            pass

    def make_child(self, parent, name):
        dirname = name
        child_path = parent + '/' + dirname
        try:
            os.mkdir(child_path)
        except FileExistsError: pass
        return child_path

    def make_child_i(self, parent, name):
        last_ind = self.get_last(parent)
        if last_ind is not None:
            self.sess_index = last_ind + 1
        dirname = name+'_{}'.format(self.sess_index)
        child_path = parent + '/' + dirname
        os.mkdir(child_path)
        self.logs_child_path = child_path
        return child_path

    def get_last(self, parent):
        contents = os.listdir(parent)
        inds = [int(os.path.splitext(x)[0].split(sep='/')[-1].split(sep='_')[-1]) for x in contents]
        try:
            return sorted(inds).pop()
        except IndexError:
            return None