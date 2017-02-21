from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
from tabulate import tabulate

from RNN.classes.RNNModels import Basic_LSTM_Model, Basic_RNN_Model
from RNN.classes.Logger import Logger
from RNN import reader

from utilities.make_table import make_table
from PDPATH import PDPATH

logging = tf.logging

def data_type():
  return tf.float32


def get_config():
    return Configs()


class Configs(object):
    init_scale = 0.1
    learning_rate = 0.05
    max_grad_norm = 5
    num_layers = 1
    num_steps = 3
    hidden_size = 12
    keep_prob = 1
    lr_decay = 1
    batch_size = 4
    vocab_size = 8


class TestData(object):
    """The input data."""
    def __init__(self, config, data, testset=False, name=None):
        # if testset:
        #   do something different for input_data and targets
        #       - We can discard targets and just look at relative ratios
        self.vocab_size = config.vocab_size
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.enqueuer(
            data, batch_size, num_steps, name=name)


def run_test():
    pass


def main(_):
    path = PDPATH('/RNN/data/tiny_data')
    raw_data = reader.raw_data(path)
    train_data, valid_data, test_data, _ = raw_data

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 4
    eval_config.num_steps = 3

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale,
                                                    seed = 1)

        with tf.name_scope("Test"):
          test_input = TestData(config=eval_config, data=test_data, name="TestInput")
          with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = Basic_RNN_Model(is_training=False, config=eval_config, input_=test_input)

        logger = Logger()
        sv = tf.train.Supervisor(logdir = logger.child_path)
        perp_test = []
        out = []
        with sv.managed_session() as session:
            test_perplexity, outputs = run_test(session, mtest)
            perp_test.append(test_perplexity)
            out.append(outputs)


if __name__ == "__main__": tf.app.run()