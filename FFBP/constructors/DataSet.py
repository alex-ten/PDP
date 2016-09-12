# CPR
import numpy as np
from numpy import ndarray

class DataSet(object):
    def __init__(self, filename):
        self.images, self.labels, self.names = self._load_txt(filename)
        self.image_size = len(self.images[0])
        self._num_examples = len(self.images)
        self._epochs_completed = 0
        self._marker_index = 0

    def permute(self):
        if self._marker_index == self._num_examples or self._marker_index==0:
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
        return(self.images[start:end], self.labels[start:end])

    def get_batch(self, inds):
        return self.images[inds]

    def get_name(self, p):
        if type(p) is list:
            p = np.array(p)
        inds, = np.where(np.all(self.images == p, axis=1))
        bucket = []
        for i in inds:
            bucket.append(self.names[i])
        if len(bucket) > 1:
            return bucket
        else:
            return bucket[0]

    def get_img(self, s):
        return self.images[self.names.index(s)]

    def split_set(self,train,valid,test):
        pass

    def _load_txt(self, filename):
        imgs = []
        lbls = []
        names = []
        for numline, line in enumerate(open(filename)):
            if len(line) <= 1: continue
            if numline == 0:
                continue
            else:
                l = line.split(sep=',')
                imgs.append([int(x) for x in l[1].split()])
                names.append(Pattern(l[0].split()[0], imgs[-1]))
                lbls.append([int(x) for x in l[2].split()])
        imgs_np_array = np.array(imgs)
        lbls_np_array = np.array(lbls)
        return imgs_np_array, lbls_np_array, names

class Pattern():
    def __init__(self, name, vect):
        self.name = name
        self.vect = vect
    def __str__(self):
        return self.name
