from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from SRN.sandbox import reader
from classes.RNNModels import Basic_LSTM_Model, Basic_RNN_Model
logging = tf.logging

def data_type():
  return tf.float32


class InputData(object):
    """The input data."""
    def __init__(self, config, data, name=None):
        self.vocab_size = config.vocab_size
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.enqueuer(
            data, batch_size, num_steps, name=name)


class Configs(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 0.05
    max_grad_norm = 5
    num_layers = 3
    num_steps = 3
    hidden_size = 10
    max_epoch = 5
    max_max_epoch = 1000
    keep_prob = 1.0
    lr_decay = 1
    batch_size = 4
    vocab_size = 8


def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""

    # Clean initialization
    start_time = time.time()
    costs = 0.0
    iters = 0
    output = None
    state = session.run(model.initial_state)
    # Values to extract from running the graph (cost, final state, and may be eval_op)
    fetches = {'cost': model.cost,
               'final_state': model.final_state}
    if eval_op is not None:
        fetches["eval_op"] = eval_op
    if not model.is_training:
        fetches['output'] = model.seq_outputs

    # Run the model epoch_size times
    for step in range(model.input.epoch_size):
        feed_dict = {}
        try:
            for i, (c, h) in enumerate(model.initial_state):
                # feed zero values to the model's initial state if state is a tuple
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h
        except TypeError:
            feed_dict[model.initial_state] = state

        # Evaluate fetches by running the graph
        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]
        if 'output' in vals.keys():
            output = vals['output']


        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
            iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters), output


def get_config():
    return Configs()


def main(_):
    path = '/Users/alexten/Projects/PDP/SRN/sandbox/simple-examples/tiny_data'

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
        with tf.name_scope("Train"):
            train_input = InputData(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = Basic_LSTM_Model(is_training=True, config=config, input_=train_input)

        with tf.name_scope("Test"):
          test_input = InputData(config=eval_config, data=test_data, name="TestInput")
          with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = Basic_LSTM_Model(is_training=False, config=eval_config, input_=test_input)

        sv = tf.train.Supervisor(logdir=path+'/junk')
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 1)
                m.assign_lr(session, config.learning_rate * lr_decay)
                # print("Epoch: %d Learning rate: %.3f" % (i, session.run(m.lr)))
                train_perplexity, _ = run_epoch(session, m, eval_op=m.train_op, verbose=False)

                if i % (config.max_max_epoch // 100) == 0:
                    print("Epoch: {} Perplexity = {}".format(i, train_perplexity))
                    test_perplexity, outputs = run_epoch(session, mtest)
                    np.set_printoptions(precision=2, suppress=True)
                    print(outputs)



if __name__ == "__main__": tf.app.run()