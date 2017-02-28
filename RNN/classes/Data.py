import numpy as np
from RNN.reader import enqueuer, get_vocab, Vocab

class InputData(object):
    """The input data."""
    def __init__(self, config, data, name=None):
        # if testset:
        #   do something different for input_data and targets
        #       - We can discard targets and just look at relative ratios
        self.vocab_size = config.vocab_size
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = enqueuer(data, batch_size, num_steps, name=name)


class TestData(object):
    """The input data for tester"""
    def __init__(self, config, test_data, test_meta, vocab, name=None):
        data = test_data
        max_seq_len = max([len(seq) for seq in data])
        self.meta = test_meta
        self.vocab = vocab
        self.vocab_size = config.vocab_size
        self.num_steps = max_seq_len - 1
        self.batch_size = len(data)
        self.epoch_size = 1
        test_inp = np.zeros((self.batch_size, self.num_steps), dtype=int) + vocab.getid(vocab.eos)
        test_targ = np.zeros((self.batch_size, self.num_steps), dtype=int) + vocab.getid(vocab.eos)

        for irow, trow, seq in zip(test_inp, test_targ, data):
            irow[0:len(seq)-1] = np.array(seq[0:-1]).astype(int)
            trow[0:len(seq)-1] = np.array(seq[1:]).astype(int)
        self.input_data = test_inp
        self.targets = test_targ

def main():
    import RNN.reader as reader
    from RNN.trainer import TinyConfigs
    from PDPATH import PDPATH

    ptb_vocab = get_vocab('ptb.voc')
    raw_test_data = reader.make_test(PDPATH('/RNN/test_data/coffee.txt'), ptb_vocab)

    test_input = TestData(config=TinyConfigs(),
                          test_data=raw_test_data,
                          vocab=ptb_vocab,
                          name="TestInput")


if __name__=='__main__': main()