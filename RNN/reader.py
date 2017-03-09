"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import pickle
import os

import tensorflow as tf
from PDPATH import PDPATH

# additions to tensorflow tutorial (see below)
class Vocab(object):
    def __init__(self, s2id, end_of_sequence='<eos>', unknown_string = '<unk>'):
        self.eos = end_of_sequence
        self.unk = unknown_string
        self.s2id = s2id
        self.id2s = {}
        for k,v in s2id.items():
            self.id2s[v] = k

        if not self.unk in list(self.s2id.keys()):
            unk_id = max(list(self.id2s.keys())) + 1
            self.s2id[self.unk] = unk_id
            self.id2s[unk_id] = self.unk

    def getid(self, s):
        try:
            return self.s2id[s]
        except KeyError:
            try:
                return self.s2id[self.unk]
            except KeyError:
                return

    def gets(self, id):
        try:
            return self.id2s[id]
        except KeyError:
            print('ID {} is not in vocab'.format(id))


def _read_raw_test(filename):
    # reads test from file and splits each line into context (key) and targets (list of values)
    r = collections.OrderedDict()
    with tf.gfile.GFile(filename, 'r') as f:
        for l in f.readlines():
            context, items = l.split(sep='%')
            items = items.replace('\n', '')            # Remove the newline character
            items = items.replace(' ', '')             # Remove whitespaces
            r[context] = items.split(sep=',')          # Split by separator (store targs by context in dict)
    return r


def _test_to_ids(test, vocab):
    # converts a test dict to word ids from vocab and returns a list of items
    # each item is a list of test items in their contexts
    # maxlen is returned for padding
    ids = []
    meta = []
    unks = []
    for context, items in test.items():
        # Include meta information (length of context and number of targets)
        cont_len = len(context.split())
        num_targs = len(items)
        meta.append((cont_len, num_targs))

        # Convert test dict to ids from vocab
        test_seqs = [(context + i + ' ' + vocab.eos) for i in items]
        for seq in test_seqs:
            spl = seq.split()
            [unks.append(i) for i in spl if vocab.getid(i) == vocab.getid(vocab.unk)]
            converted_seq = [vocab.getid(i) for i in spl]
            ids.append(converted_seq)
    return ids, meta, unks


def make_test(filename, vocab):
    print('Preparing test...')
    test_dict = _read_raw_test(filename)
    converted_test, meta, unks = _test_to_ids(test_dict, vocab)
    print('Found {} unknown items:'.format(len(unks)))
    for unk in unks:
        print('  * {}'.format(unk))
    return converted_test, meta


def get_vocab(filename):
    # unpickle Vocab
    f = PDPATH('/RNN/vocabs/{}'.format(filename))
    v = pickle.load(open(f, 'rb'))
    return v


# reader functions from tensorflow tutorial
def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    # Return a long list of word sequences with <eos> in-between individual sentences
    return f.read().decode('utf-8').replace("\n", "<eos>").split()


def _build_vocab(filename, sorted_words_only=False):
    # Long list of word sequences separated by <eos>
    data = _read_words(filename)
    # Stores tallies of unique words in data, e.g. {''<unk>': 4794, 'the': 4529, '<eos>': 3761}
    counter = collections.Counter(data)
    # Creates an ordered list of 2-tuples containing a WORD and its TALLY
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0])) # x[0] is a backup criterion in case -x[1] are equal
    if sorted_words_only:
        return [x[0] for x in count_pairs]
    # Creates a tuple of words sorted in descending order from most frequent to least frequent
    words, _ = list(zip(*count_pairs))
    # Assign a unique integer ID to each word
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id # Return a dict with unique words as keys and their ID as values


def _file_to_word_ids(filename, word_to_id):
    # Returns an INDEXED version of the original file. Each word is replaced with an index
    # Indices correspond to frequency of occurance in the corpus (0 means the most frequent index)
    data = _read_words(filename)
    r = []

    for word in data:
        if word in word_to_id:
            r.append(word_to_id[word])
        else:
            print('Not in list: {}'.format(word))
    return r


def raw_data(data_path=None):
  """Load raw data from data directory "data_path".
  Reads text files, converts strings to integer ids,
  and performs mini-batching of the inputs.
  Args:
    data_path: string path to the directory where data has
      been extracted.
  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "train.txt")
  valid_path = os.path.join(data_path, "valid.txt")
  test_path = os.path.join(data_path, "test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary


def enqueuer(raw_data, batch_size, num_steps, name=None):
    """Iterate on the raw PTB data.
    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.
    Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).
    Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.
    Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    with tf.name_scope(name, "InputEnqueuer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name = "raw_data", dtype = tf.int32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(epoch_size, message = "epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
        epoch_size = tf.identity(epoch_size, name = "epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue() # output queue index from the queue object (relies on tf.train.Supervisor)
    x = tf.slice(data, [0, i * num_steps], [batch_size, num_steps]) # slice data
    y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps])
    return x, y

# main scripts
def reader_demo():
    file = PDPATH('/RNN/test_data/ptb_word_data/test.txt')
    print('Step 1. Convert raw corpus into a long list:')
    print(_read_words(file))

    print('\nStep 2. Build vocab (assign strings to IDs):')
    print(_build_vocab(file))

    print('\nStep 3. Convert words into word IDs:')
    print(_file_to_word_ids(file, _build_vocab(file)))

    print('\nAdd step. Sort unique words by frequency or alphabetically:')
    print(_build_vocab(file, True))

def vocab_demo():
    v = get_vocab('ptb.voc')
    items = ['the', 'dog', 'dogs', 'boy', 'boys', 'is','are','has','have','was','were']
    for i in items:
        print(i, v.getid(i))

def make_vocab():
    file = PDPATH('/RNN/train_data/tiny_data/train.txt')
    s2id = _build_vocab(file)
    V = Vocab(s2id)
    pickle.dump(V, open(PDPATH('/RNN/vocabs/tiny.voc'), 'wb'))


def sandbox():
    def f(filename, sorted_words_only=False):
        # Long list of word sequences separated by <eos>
        data = _read_words(filename)
        # Stores tallies of unique words in data, e.g. {''<unk>': 4794, 'the': 4529, '<eos>': 3761}
        counter = collections.Counter(data)
        return counter
    file = PDPATH('/RNN/train_data/ptb_word_data/train.txt')
    d=f(file)

    items = ['the', 'dog', 'dogs', 'boy', 'boys', 'is', 'are', 'has', 'have', 'was', 'were']
    for i in items:
        print(i, d[i])

if __name__=='__main__': sandbox()