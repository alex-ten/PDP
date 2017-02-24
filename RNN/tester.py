from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import code # todo remove after development
import numpy as np
import tensorflow as tf

from utilities.make_table import make_table
from PDPATH import PDPATH

from RNN.classes.RNN_Models import Basic_LSTM_Model, Basic_RNN_Model
from RNN.classes.Logger import Logger
from RNN import reader
from RNN.trainer import run_epoch
flags = tf.flags
logging = tf.logging
flags.DEFINE_string("m_path", None, "Path to LM .ckpt files.")

FLAGS = flags.FLAGS


def data_type():
  return tf.float32


def get_config():
    return Configs()


class Configs(object):
    num_layers = 1
    num_steps = 3
    hidden_size = 20
    keep_prob = 1
    lr_decay = 0.8
    batch_size = 4
    vocab_size = 8
    max_grad_norm=1
    init_scale = 0.05
    learning_rate = 0.025
    max_epoch = 1000
    max_max_epoch = 1000


def run_test(session, model):
    print('I am supposed to run a test, but I don\'t know how :(')


def main(_):
    path = PDPATH('/RNN/train_data/tiny_data')
    raw_data = reader.raw_data(path)
    train_data, valid_data, test_data, _ = raw_data

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 4
    eval_config.num_steps = 3

    with tf.Graph().as_default() as graph:
        with tf.Session() as session:
            test_input = TestData(config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model"):
                mtest = Basic_RNN_Model(is_training=False, config=eval_config, input_=test_input)

            saver = tf.train.Saver()
            saver.restore(session, PDPATH('/RNN/trained_models/') + FLAGS.m_path + '')

            a,b = run_epoch(session=session, model=mtest)
            np.set_printoptions(2,suppress=True)
            print(np.around(b,2))


if __name__ == "__main__": tf.app.run()