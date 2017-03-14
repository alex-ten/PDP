import os
import pickle
import numpy as np


from PDPATH import PDPATH

class NetworkData(object):
    def __init__(self, path):
        self.path = path.replace('snap.pkl','')
        with open(path, 'rb') as opened_file:
            snapshot = pickle.load(opened_file)
        self.error = snapshot.pop('error')
        self.epochs = snapshot.pop('epochs')
        self.attributes = snapshot.pop('attributes')
        self.inp_names = snapshot.pop('inp_names')
        self.inp_vects = snapshot.pop('inp_vects')
        self.hyperparams = snapshot.pop('hyperparams')
        self.sess_index = snapshot.pop('sess_index')
        self.main = snapshot
        self.layer_names = self.main.keys()
        self.num_layers = len(self.main)
        self.num_units = sum([x.size for x in self.main.values()])

    def summary(self):
        print('')
        np.set_printoptions(precision=3, suppress=True, linewidth=200)
        print('==' * 10 + ' SNAPSHOT SUMMARY ' + '==' * 10, end='\n\n')
        print('>>> Error:')
        print('    {}'.format(self.error), end='\n\n')
        for layer, log in self.main.items():
            print('>>> ' + layer + ':', end='\n\n')
            for k, v in log.__dict__.items():
                if k=='name' or k=='size' or k=='sender' or len(v)==0: continue
                a = len(k)
                print('    ' + '{}:'.format(k) +
                      ' '*(20-a) +
                      '{} states of {}x{} array'.format(len(v),
                                                        np.shape(v[0])[0],
                                                        np.shape(v[0])[1]))
            print('\n')
        print('===' * 20, '')

    def scsv(self, filename, layer, tind, variable):
        if filename in os.listdir(self.path):
            permit = 'ab'
        else:
            permit = 'wb'
        a = self.main[layer].__dict__[variable][tind]
        file = open(self.path + filename, mode=permit)
        np.savetxt(file, a, delimiter=',')
        file.close()


def main():
    proceed = True
    usrdir = input('[FFBP Viewer] Provide user directory (if any), or press \'enter\' to use default directory: ')
    usrdir = usrdir.strip()
    while proceed:
        path = input('[FFBP Viewer] Enter name of log directory OR corresponding index: ')
        # Get path to snap file
        try:
            ID = int(path)
            path = PDPATH('/FFBP{}/logs/FFBPlog_{}/snap.pkl'.format('/' + usrdir if len(usrdir) else '', path))
        except ValueError:
            ID = int(path.split(sep='_')[-1])
            path = PDPATH('/FFBP{}/logs/'.format('/' + usrdir if len(usrdir) else '') + path + '/snap.pkl')
        with open(path, 'rb'):
            snap = NetworkData(path)
            snap.summary()
            print(snap.main['hidden1'])

        print('[FFBP Viewer] Would you like to proceed?')
        prompt = input("[y/n] -> ")
        if prompt == 'n': proceed = False


if __name__=='__main__': main()