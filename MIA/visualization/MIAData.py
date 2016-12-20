import pickle
import numpy as np

class MIAData():
    def __init__(self, path):
        dims = path.split(sep='snaplog--').pop(1).split(sep='.').pop(0).split('-')
        with open(path, 'rb') as file:
            log = pickle.load(file)
        self.snaps = np.arange(len(log)) # np.array containing snap inds
        self.main = log
        self.data_dim = float(dims[0])
        self.max_len = float(dims[1])
        self.hid_size = float(dims[2])
        self.num_epochs = len(log)

    def __getitem__(self, item):
        return self.main[item]

def main():
    np.set_printoptions(suppress=False, linewidth=1000, threshold=500,edgeitems=500)
    logpath = '/Users/alexten/Projects/PDP/SRN/logdir/Sess_2016-10-24_20-57-34/mpl_data/snaplog--7-31-3.pkl'
    data = RNData(logpath)

    print(data.snaps, data.data_dim, data.max_len, data.hid_size)
    print(len(data.main))
    for k,v in data.main[0].items():
        print(k, np.shape(v))

    snap = data[0]
    pattern_ind = 3
    max_len = 31

    a = pattern_ind * max_len
    b = a + snap['seq_lens'][pattern_ind] - 1
    inp_sequence = snap['inp'][a:b]
    hid_sequence = snap['hid'][a:b]
    out_sequence = snap['out'][a:b]
    targ_sequence = snap['targ'][a:b]
    print(inp_sequence)
    print(hid_sequence)
    print(out_sequence)
    print(targ_sequence)

if __name__=='__main__': main()

