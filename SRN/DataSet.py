import numpy as np
from collections import OrderedDict
import pickle
from time import sleep

def make_oh_map(chars):
    map_ = OrderedDict()
    num_chars = len(chars)
    for ind, char in enumerate(sorted(chars)):
        z = np.zeros(num_chars)
        z[ind] = 1.0
        map_[char] = z
    map_['_length_'] = num_chars
    return map_


def str2oh(raw, max_length, oh_map, NaN_fill = False):
    num_cols = oh_map['_length_']
    num_rows = max_length# - 1 #todo inserted -1
    num_planes = len(raw)
    sequence_x = np.zeros((num_planes, num_rows, num_cols))
    sequence_y = np.zeros((num_planes, num_rows, num_cols))
    if NaN_fill:
        sequence_x[:] = np.NaN
        sequence_y[:] = np.NaN
    for plane_ind, sequence in enumerate(raw):
        for row_ind, char in enumerate(sequence):
            #if row_ind == num_rows: break #todo added this whole line
            sequence_x[plane_ind,row_ind] = oh_map[char]
        sequence_y[plane_ind,0:-1] = sequence_x[plane_ind,1:]

    #return sequence_x[:, 0:-1], sequence_y[:, 0:-1]
    return sequence_x, sequence_y


class DataSet(object):
    def __init__(self, path):
        with open(path, 'rb') as file:
            jar = pickle.load(file)
            self.unique = jar.pop()
            self.raw = jar
        self.num_seqs = len(self.raw)
        self.oh_map = make_oh_map(self.unique)
        self.max_length = len(max(self.raw, key=len))
        self.inp_seqs = None
        self.out_seqs = None
        self._current_batches = None
        self._current_batches_size = None
        self._batches_ind = 0
        self._batch_ind = 0


    def raw2onehot(self, NaN_fill = False):
        self.inp_seqs, self.out_seqs = str2oh(raw = self.raw,
                                              max_length = self.max_length,
                                              oh_map = self.oh_map,
                                              NaN_fill = NaN_fill)

    def _get_batches(self, num_seqs):
        assert self.num_seqs % num_seqs == 0, 'Number of batches must divide the total ' \
                                                 'number of sequences. Got {} % {} = {}'.format(self.max_length,
                                                                                                num_seqs,
                                                                                                self.max_length % num_seqs)
        start = self._batches_ind
        self._batches_ind += num_seqs
        if self._batches_ind > self.num_seqs:
            start = 0
            self._batches_ind = num_seqs
        end = self._batches_ind
        _x, _y = np.squeeze(self.inp_seqs[start:end,:]), np.squeeze(self.out_seqs[start:end,:])
        zxy_x, zxy_y = np.shape(_x), np.shape(_y)

        _x = np.reshape(_x, (zxy_x[0] * zxy_x[1], zxy_x[2]), order='F')
        _y = np.reshape(_x, (zxy_y[0] * zxy_y[1], zxy_y[2]), order='F')
        self._current_batches_size = num_seqs * self.max_length
        self._current_batches = _x, _y

    def next_batch(self, batch_size):
        if not self._batch_ind:
            self._get_batches(batch_size)
        assert self._current_batches_size % batch_size == 0, 'Batch size must divide the total ' \
                                                             'number of sequences. Got {} % {} = {}'.format(self.max_length,
                                                                                                            batch_size,
                                                                                                            self.max_length % batch_size)
        start = self._batch_ind
        self._batch_ind += batch_size
        if self._batch_ind > batch_size * self.max_length:
            self._get_batches(batch_size)
            start = 0
            self._batch_ind = batch_size
        end = self._batch_ind
        xs, ys = self._current_batches
        return xs[start:end], ys[start:end]

    def all_seqs(self):
        return (self.inp_seqs, self.out_seqs)


def main():
    data = DataSet('240.pkl')
    data.raw2onehot()
    print(data.raw)

    for i in range(8):
        x, y = data.next_batch(2)

if __name__=='__main__': main()