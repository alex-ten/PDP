from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import pickle
import numpy as np
import tensorflow as tf

from RNN.classes.RNN_Models import Basic_LSTM_Model, Basic_RNN_Model
from RNN.classes.Data import InputData
from RNN.classes.Logger import Logger
from RNN.classes.Configs import Configs
from RNN import reader

from utilities.printProgress import printProgress
from utilities.save_plot import save_plot
from PDPATH import PDPATH

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("name", None, "Model name. If none, directories will be named with default names")
flags.DEFINE_string("train_data", None, "Training data directory (must contain .train, .test, .valid .txt files).")
flags.DEFINE_string("save_as", None, "Model output directory.")
flags.DEFINE_bool("use_fp16", False, "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool("prog", False, "Show progress bar in stdout (for interactive usage).")

FLAGS = flags.FLAGS


def print_(i,t,v):
    print("Epoch: {}    Train perplexity = {}   Validation perplexity: {}".format(i,t,v))


def get_config():
    return Configs()


def get_model(cell, is_training, **kwargs):
    if cell == 'LSTM':
        return Basic_LSTM_Model(is_training=is_training, config=kwargs['config'], input_=kwargs['input_'])
    elif cell == 'RNN':
        return Basic_RNN_Model(is_training=is_training, config=kwargs['config'], input_=kwargs['input_'], BPTT=True)
    elif cell == 'SRN':
        return Basic_RNN_Model(is_training=is_training, config=kwargs['config'], input_=kwargs['input_'], BPTT=False)
    return


def save_config(c, filename):
    # Pickle a Configs object with .config extension
    # Also save configs as a txt file
    pickle.dump(c, open(filename+'.config', 'wb'))
    with open(filename+'.config.txt', 'w') as txt:
        txt.write('class Configs(object):\n')
        for i in [attr for attr in dir(c) if not attr.startswith('__')]:
            txt.write('    {} = {}\n'.format(i, c.__getattribute__(i)))


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

    config = Configs(batch_size = 20,
                    hidden_size = 1500,
                    init_scale = 0.04,
                    keep_prob = 0.35,
                    learning_rate = 1.0,
                    lr_decay = 1/1.15,
                    max_epoch = 14,
                    max_grad_norm = 10,
                    max_max_epoch = 55,
                    model = 'LSTM',      # Set of available models: 'LSTM', 'RNN', 'SRN'
                    num_layers = 2,
                    num_steps = 35,
                    vocab_size = 10000)
    eval_config = config.clone()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    if FLAGS.train_data: path = PDPATH('/RNN/train_data/'+FLAGS.train_data)
    else:
        print('Provide path to training data, e.g: train.py --train_data=\'path\'')
        return

    logger = Logger()

    raw_data = reader.raw_data(path)
    train_data, valid_data, test_data, _ = raw_data

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                     config.init_scale,
                                                     seed = 1)
        with tf.name_scope("Train"):
            train_input = InputData(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = get_model(config.model, is_training=True, config=config, input_=train_input)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            valid_input = InputData(config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = get_model(config.model, is_training=False, config=config, input_=valid_input)
            tf.summary.scalar("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            test_input = InputData(config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = get_model(config.model, is_training=False, config=eval_config, input_=test_input)

        logger.make_child_i(logger.logs_path, 'RNNlog')
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Model'),
                               sharded=False,
                               write_version=tf.train.SaverDef.V2)
        sv = tf.train.Supervisor(logdir = logger.logs_child_path, saver=saver)
        train_log = []
        valid_log = []
        out = []

        # Session runs here
        # Setup session configs
        sess_config = tf.ConfigProto(log_device_placement=False)
        sess_config.gpu_options.allow_growth = True
        # Start session context manager by calling to tf.train.Supervisor's managed_session
        with sv.managed_session(config=sess_config) as session:
            if FLAGS.prog: printProgress(0, config.max_max_epoch, 'Training', 'Complete', barLength=60)
            for i in range(config.max_max_epoch):
                fin = i + 1
                valid_perplexity, _ = run_epoch(session, mvalid)
                valid_log.append(valid_perplexity)
                if len(valid_log) >= 2:
                    if valid_log[-1] > valid_log[-2]:
                        break

                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 1)
                m.assign_lr(session, config.learning_rate * lr_decay)

                train_perplexity, _ = run_epoch(session, m, eval_op=m.train_op)
                train_log.append(train_perplexity)

                output_frequency = 5
                if config.max_max_epoch >= output_frequency:
                    if (i % (config.max_max_epoch // output_frequency) == 0) or i==config.max_max_epoch-1:
                        print_(i, train_perplexity, valid_perplexity)
                else:
                    print_(i, train_perplexity, valid_perplexity)

                if FLAGS.prog:
                    printProgress(i+1, config.max_max_epoch, 'Training', 'Complete', barLength=60)

            test_perplexity, outputs = run_epoch(session, mtest)
            print('\nStopped training on epoch {}'.format(fin))
            print("Test perplexity: {}".format(test_perplexity))

            if FLAGS.save_as:
                if FLAGS.name:
                    save_to = logger.make_child(logger.trained_path, FLAGS.name)
                else:
                    save_to = logger.make_child_i(logger.trained_path, 'model')

                spath = save_to +'/'+ FLAGS.save_as
                print("\nSaving model to {}.".format(spath))
                saver.save(session, spath, global_step=sv.global_step)
                save_config(config, filename=spath)
                save_plot('Learning curves from {}'.format(FLAGS.save_as), save_to, train_log, valid_log)


if __name__ == "__main__": tf.app.run()