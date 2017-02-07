import pickle
import numpy as np
import code
class NetworkData(object):
    def __init__(self, path):
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

    def stdout(self):
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


def main():
    path = input('Enter path to snapshot: ')
    data = NetworkData(path)
    print(data.epochs)
    print(data.error)
    print(data.inp_names)
    print(data.inp_vects)
    print(data.hyperparams)
    # data.stdout()

if __name__=='__main__': main()