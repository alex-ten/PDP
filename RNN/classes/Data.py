from RNN.reader import enqueuer

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
    def __init__(self, config, data, testset=False, name=None):
        # if testset:
        #   do something different for input_data and targets
        #       - We can discard targets and just look at relative ratios
        self.vocab_size = config.vocab_size
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = [[1,2,3],[1,6,3],[4,2,5],[4,7,5]], [[2,3,0],[6,3,0],[2,5,0],[7,5,0]]