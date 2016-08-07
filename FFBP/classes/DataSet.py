# CPR
import random
import numpy as np

class DataSet(object):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.image_size = len(images[0])
        self._num_examples = len(images)
        self._epochs_completed = 0
        self._marker_index = 0
    def permute(self):
        if self._marker_index == self._num_examples or self._marker_index==0:
            rng_state = np.random.get_state()
            np.random.shuffle(self.images)
            np.random.set_state(rng_state)
            np.random.shuffle(self.labels)
        else: pass
    def next_batch(self, batch_size):
        assert self._num_examples % batch_size == 0
        start = self._marker_index
        self._marker_index += batch_size
        if self._marker_index > self._num_examples:
            start = 0
            self._marker_index = batch_size
        end = self._marker_index
        return(self.images[start:end], self.labels[start:end])
    def split_set(self,train,valid,test):
        pass

def load_data(filename):
    imgs = []
    lbls = []
    for numline, line in enumerate(open(filename)):
        if len(line) <= 1: continue
        if numline == 0:
            i_ind = line.find('input')
            o_ind = line.rfind('output')
        else:
            imgs.append([int(x) for x in line[i_ind:o_ind].split()])
            lbls.append([int(x) for x in line[o_ind:].split()])
    imgs_np_array = np.array(imgs)
    lbls_np_array = np.array(lbls)
    return DataSet(imgs_np_array, lbls_np_array)