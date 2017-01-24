import os
import pickle

class Logger():
    def __init__(self):
        self.parent_path = os.getcwd() + '/MIA/logs'
        self.may_be_make_parent()

    def may_be_make_parent(self):
        try:
            os.mkdir(self.parent_path)
        except FileExistsError:
            pass

    def get_last(self):
        contents = os.listdir(self.parent_path)
        inds = [int(os.path.splitext(x)[0].split(sep='/')[-1].split(sep='_')[-1]) for x in contents]
        try:
            return sorted(inds).pop()
        except IndexError:
            return None

    def save(self, log):
        last_ind = self.get_last()
        if last_ind is None:
            filename = 'MIAlog_0.pkl'
        else:
            filename = 'MIAlog_{}.pkl'.format(last_ind+1)
        with open(self.parent_path + '/' + filename, 'wb') as new_file:
            pickle.dump(log, new_file)
        return self.parent_path + '/' + filename