import numpy as np
from collections import OrderedDict
import pickle
import random

def make_oh_map(chars):
    map_ = OrderedDict()
    num_chars = len(chars)
    for ind, char in enumerate(sorted(chars)):
        z = np.zeros(num_chars)
        z[ind] = 1.0
        map_[char] = z
    map_['_length_'] = num_chars
    return map_


class DataSet(object):
    def __init__(self, path):
        with open(path, 'rb') as file:
            jar = pickle.load(file)
            self.unique = sorted(jar.pop())
            self.raw = jar
        self.num_seqs = len(self.raw)
        self.oh_map = make_oh_map(self.unique)
        self.max_length = len(max(self.raw, key=len))
        self.seq_lengths = np.array([len(r) for r in self.raw])
        self.inp_seqs = None
        self.out_seqs = None
        self.inp_inds = None
        self.out_inds = None
        self._current_batches = None
        self._current_batches_size = None
        self._batches_ind = 0
        self._batch_ind = 0

    def raw2onehot(self, NaN_fill = False):
        num_cols = self.oh_map['_length_']
        num_rows = self.max_length - 1
        num_planes = len(self.raw)
        self.inp_seqs = np.zeros((num_planes, num_rows, num_cols))
        self.out_seqs = np.zeros((num_planes, num_rows, num_cols))
        if NaN_fill:
            self.inp_seqs[:] = np.NaN
            self.out_seqs[:] = np.NaN
        for plane_ind, sequence in enumerate(self.raw):
            l = len(sequence)
            seq_placeholder = np.zeros((l, num_cols))
            for row_ind, char in enumerate(sequence):
                seq_placeholder[row_ind] = self.oh_map[char]
            self.inp_seqs[plane_ind][:l-1] = seq_placeholder[:-1]
            self.out_seqs[plane_ind][:l-1] = seq_placeholder[1:]

    def raw2inds(self):
        self.inp_inds = np.zeros(shape = [self.num_seqs, self.max_length - 1])
        self.out_inds = np.zeros(shape = [self.num_seqs, self.max_length - 1])
        for row, sequence in enumerate(self.raw):
            l = len(sequence)
            seq_placeholder = np.zeros(l)
            for i, char in enumerate(sequence):
                seq_placeholder[i] = self.unique.index(char)
            self.inp_inds[row][0:l-1] = seq_placeholder[:-1]
            self.out_inds[row][0:l-1] = seq_placeholder[1:]


    def next_batch(self, batch_size, ind_batch_X = False, ind_batch_Y = False):
        if self.num_seqs % batch_size != 0:
            raise ValueError('Batch size must divide the total number of sequences in the dataset')
        start = self._batch_ind
        self._batch_ind += batch_size
        if self._batch_ind > self.num_seqs:
            start = 0
            self._batch_ind = batch_size
        end = self._batch_ind
        _x, _y = self.inp_seqs[start:end,:,:], self.out_seqs[start:end,:,:]
        _l = self.seq_lengths[start:end]
        _s = self.raw[start:end]
        if ind_batch_X:
            _x = self.inp_inds[start:end].astype(int)
        if ind_batch_Y:
            _y = self.out_inds[start:end].astype(int)
        return _x, _y, _l, _s

    def all_seqs(self):
        return (self.inp_seqs, self.out_seqs)

    def get_oh(self, s):
        return self.oh_map[s]

    def show_oh_map(self):
        _map = self.oh_map
        length = _map.pop('_length_')
        print('vocab_size = {}'.format(length))
        for k, v in self.oh_map.items():
            print(k, v)

def main():
    data = DataSet('/Users/alexten/Projects/PDP/RRN/pickles/a_through_x.pkl')
    data.raw2onehot()
    data.raw2inds()


    if len(data.raw) > 20:
        print('raw contains {} sequences. You shouldn\'t display so much output. Here\'s a sample of 15 items:'.format(len(data.raw)))
        sample = random.sample(data.raw, 10)
        for s in sample:
            print('{0:4}  {1}'.format(data.raw.index(s), s, align='left'))
    else:
        for i, s in enumerate(data.raw):
            print('{0:4}  {1}'.format(i,s))

    data.show_oh_map()
    for i in range(1):
        x, y, _, __ = data.next_batch(2)
        print(x)
        print(y)


if __name__=='__main__': main()