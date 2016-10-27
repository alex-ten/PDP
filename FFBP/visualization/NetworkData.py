import pickle
import numpy as np

class NetworkData(object):
    def __init__(self, path):
        with open(path, 'rb') as opened_file:
            snapshot = pickle.load(opened_file)
        self.error = snapshot['error'] # use dict.pop() instead of del dict[]
        self.epochs = snapshot['epochs']
        self.attributes = snapshot['attributes']
        self.inp_names = snapshot['inp_names']
        self.inp_vects = snapshot['inp_vects']
        del snapshot['error']
        del snapshot['epochs']
        del snapshot['attributes']
        del snapshot['inp_names']
        del snapshot['inp_vects']
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
    # data.stdout()

if __name__=='__main__': main()