# CPR
import numpy as np
from collections import OrderedDict

class DataSet(object):
    def __init__(self, filename, datatype=float):
        self.images, self.labels, self.names = self._load_txt(filename, datatype)
        self.image_size = len(self.images[0])
        self._num_examples = len(self.images)
        self._epochs_completed = 0
        self._marker_index = 0

    def permute(self):
        if self._marker_index == self._num_examples or self._marker_index == 0:
            rng_state = np.random.get_state()
            np.random.shuffle(self.images)
            np.random.set_state(rng_state)
            np.random.shuffle(self.labels)

    def next_batch(self, batch_size):
        assert self._num_examples % batch_size == 0
        start = self._marker_index
        self._marker_index += batch_size
        if self._marker_index > self._num_examples:
            start = 0
            self._marker_index = batch_size
        end = self._marker_index
        return (self.images[start:end], self.labels[start:end])

    def big_batch(self):
        return (self.images, self.labels)

    def get_batch(self, inds):
        return self.images[inds]

    def get_name(self, p):
        if type(p) is list:
            p = np.array(p)
        inds, = np.where(np.all(self.images == p, axis=1))
        bucket = []
        for i in inds:
            bucket.append(list(self.names.keys())[i])
        if len(bucket) > 1:
            return bucket
        else:
            try:
                return bucket[0]
            except IndexError:
                ValueError('Item is not in the set')

    def get_img(self, s):
        return self.images[list(self.names.keys()).index(s)]

    def split_set(self,train,valid,test):
        pass

    def _load_txt(self, filename, datatype):
        imgs = []
        lbls = []
        names = OrderedDict()
        for numline, line in enumerate(open(filename)):
            if len(line) <= 1: continue
            if numline == 0:
                continue
            else:
                l = line.split(sep=',')
                imgs.append([datatype(x) for x in l[1].split()])
                names[l[0].split()[0]] = imgs[-1]
                lbls.append([datatype(x) for x in l[2].split()])
        imgs_np_array = np.array(imgs)
        lbls_np_array = np.array(lbls)
        return imgs_np_array, lbls_np_array, names

def main():
    x = DataSet('/Users/alexten/Projects/PDP/FFBP/data/f_XOR.txt')
    for k,v in x.names.items(): print(k,v)

    num_epochs = 5
    num_patterns = 4
    batch_size = 2
    permute = True

    for i in range(num_epochs):
        if permute:
            x.permute()
        print('\nThis is epoch {}'.format(i))
        for j in range(int(num_patterns/batch_size)):
            print('  - minibatch {}:'.format(j))
            print(x.next_batch(batch_size)[0])


if __name__=='__main__': main()