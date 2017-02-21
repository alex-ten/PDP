from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf

from RNN.classes.RNNModels import Basic_LSTM_Model, Basic_RNN_Model
from RNN.classes.Logger import Logger
from RNN import reader

from utilities.make_table import make_table
from PDPATH import PDPATH

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


def data_type():
  return tf.float32


def get_config():
    return Configs()


class Configs(object):
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


class InputData(object):
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
        # feed zero values to a the RNN's hidden state
        # the below might be confusing without some background on LSTM's state implementation
        # For more info see: http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
        try:
            for i, (c, h) in enumerate(model.initial_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h
        except TypeError:
            feed_dict[model.initial_state] = state

        # Evaluate fetches by running the graph
        # THIS IS WHERE THE ACTION OCCURS
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
        with tf.name_scope("Train"):
            train_input = InputData(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = Basic_RNN_Model(is_training=True, config=config, input_=train_input, BPTT=False)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        with tf.name_scope("Test"):
          test_input = InputData(config=eval_config, data=test_data, name="TestInput")
          with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = Basic_RNN_Model(is_training=False, config=eval_config, input_=test_input)

        logger = Logger()
        sv = tf.train.Supervisor(logdir = logger.child_path)
        perp_train = []
        perp_test = []
        out = []
        with sv.managed_session(config=tf.ConfigProto(log_device_placement=True)) as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 1)
                m.assign_lr(session, config.learning_rate * lr_decay)
                # print("Epoch: {} Learning rate: {}".format(i, np.around(session.run(m.lr),3)))
                train_perplexity, _ = run_epoch(session, m, eval_op=m.train_op, verbose=False)
                perp_train.append(train_perplexity)

                if i % (config.max_max_epoch // 10) == 0:
                    test_perplexity, outputs = run_epoch(session, mtest)
                    print("Epoch: {}\nTrain perplexity = {}\nTest perplexity: {}".format(i, train_perplexity, test_perplexity))
                    print(make_table(a = outputs,
                                     rkeys = [y for y in reader._read_words(path+'/tiny.test.txt')[1:] if y != '<eos>'],
                                     ckeys = reader._build_vocab(path+'/tiny.test.txt', True)))
                    print(session.run(mtest._input.input_data).reshape([1,-1]))
                    print(session.run(mtest._input.targets).reshape([1,-1]))
                    print(np.argmax(outputs, axis=1))
                    perp_test.append(test_perplexity)
                    out.append(outputs)

        if FLAGS.save_path:
            print("Saving model to {}.".format(PDPATH('/RNN/')+FLAGS.save_path))
            sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__": tf.app.run()