# CPR
import numpy as np
from collections import OrderedDict
import warnings
warnings.simplefilter('always', category=UserWarning)

class DataSet(object):
    def __init__(self, filename, datatype=float):
        self.images, self.labels, self.names = self._load_txt(filename, datatype)
        self.image_size = len(self.images[0])
        self.label_size = len(self.labels[0])
        self._num_examples = len(self.images)
        self._epochs_completed = 0
        self._marker_index = 0
        self._added_items = 0
        self._dtype = datatype

    def permute(self):
        if self._marker_index == self._num_examples or self._marker_index == 0:
            rng_state = np.random.get_state()
            np.random.shuffle(self.images)
            np.random.set_state(rng_state)
            np.random.shuffle(self.labels)

    def next_batch(self, batch_size):
        modulo = self._num_examples % batch_size
        if modulo != 0:
            w = 'Number of examples is not divisible by batch size. The last {} example(s) will not be included in the next batch'.format(modulo)
            warnings.warn(w)
        start = self._marker_index
        self._marker_index += batch_size
        if self._marker_index > self._num_examples:
            start = 0
            self._marker_index = batch_size
        end = self._marker_index
        return (self.images[start:end], self.labels[start:end])

    def add_item(self, item, name=None):
        # item must be a list or a tuple of patterns (lists, tuples, or arrays of numbers) and name should be a string
        x_, y = [np.array(i).astype(self._dtype) for i in item]
        x = x_.reshape([-1, self.image_size])
        y = y.reshape([-1, self.label_size])
        self.images = np.append(self.images, x, axis=0)
        self.labels = np.append(self.labels, y, axis=0)
        self._num_examples += 1
        if name:
            if name in self.names.keys(): name = '+'+name
            self.names[name] = [k for k in x]
        else:
            self.names['*add_item{}'.format(self._added_items)] = [self._dtype(k) for k in x_]
            self._added_items += 1

    def add_items(self, itemdict):
        # itemdict should be an ordered dict
        for k, v in itemdict.items():
            self.add_item(v, name=k)

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

    def add_augment_set(self, filename):
        imgs = []
        lbls = []
        self.augment_set = OrderedDict()
        for numline, line in enumerate(open(filename)):
            if len(line) <= 1: continue
            if numline == 0:
                continue
            else:
                l = line.split(sep=',')
                imgs.append([self._dtype(x) for x in l[1].split()])
                lbls.append([self._dtype(x) for x in l[2].split()])
                self.augment_set[l[0].split()[0]] = (imgs[-1], lbls[-1])

    def augment(self, n):
        # takes n examples from the augmentation set and adds them to the training examples
        pull = [k for k in self.augment_set.keys()][0:n]
        if len(pull) == 0:
            warnings.warn('Augmentation set is empty')
            return
        self.add_items({k: self.augment_set[k] for k in pull})
        for k in pull:
            self.augment_set.pop(k)


def main():
    import code
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
    print(x._marker_index, x._added_items, x._epochs_completed)

    x.add_augment_set('/Users/alexten/Projects/PDP/FFBP/data/f_XOR.txt')
    print(x.images)
    code.interact(local=locals())


if __name__=='__main__': main()