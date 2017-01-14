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
        try:
            return sorted(contents)[-1]
        except IndexError:
            return None

    def save(self, log):
        last_file = self.get_last()
        if last_file is None:
            filename = 'MIAlog_0.pkl'
        else:
            name = os.path.splitext(last_file)[0].split(sep='_')[-1]
            filename = 'MIAlog_{}.pkl'.format(int(name)+1)
        with open(self.parent_path + '/' + filename, 'wb') as new_file:
            pickle.dump(log, new_file)
        return self.parent_path + '/' + filename