from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PDPATH import PDPATH

from RNN.classes.RNN_Models import Basic_LSTM_Model, Basic_RNN_Model
from RNN.reader import Vocab
from RNN.classes.Data import TestData
from RNN import reader
from RNN.trainer import run_epoch, TinyConfigs

flags = tf.flags
logging = tf.logging
flags.DEFINE_string("model", None, "Path to trained model.")
flags.DEFINE_string("test", None, "Path to test data.")
flags.DEFINE_string("vocab", None, "Path to vocabulary")

FLAGS = flags.FLAGS


def data_type():
  return tf.float32


def load_configs(path):
    for file in os.listdir(path):
        if file.endswith('.config'):
            return pickle.load(open(os.path.join(path,file), 'rb'))


def run_test(session, model, model_input):
    perp, preds = run_epoch(session, model)
    meta = model_input.meta
    print(perp)
    print(preds)
    print(meta)
    return preds

def main(_):
    vocab = reader.get_vocab(FLAGS.vocab)
    test_ids, test_meta = reader.make_test(PDPATH('/RNN/test_data/'+FLAGS.test), vocab)
    print(test_ids)
    model_path = PDPATH('/RNN/trained_models/') + FLAGS.model
    config = load_configs(model_path)


    with tf.Graph().as_default() as graph:
        with tf.Session() as session:
            test_input = TestData(config = config,
                                  test_data = test_ids,
                                  test_meta = test_meta,
                                  vocab=vocab,
                                  name="TestInput")

            with tf.variable_scope("Model"):
                mtest = Basic_LSTM_Model(is_training=False, config=config, input_=test_input)

            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
            saved_files = os.listdir(model_path)
            for file in saved_files:
                if '.meta' in file:
                    ckpt = file.split(sep='.')[0]
                    saver.restore(session, os.path.join(model_path,ckpt))
                    continue

            b = run_test(session=session, model=mtest, model_input=test_input)
            np.set_printoptions(2,suppress=False)
            print(np.around(b,2))
            print(np.shape(b))

if __name__ == "__main__": tf.app.run()