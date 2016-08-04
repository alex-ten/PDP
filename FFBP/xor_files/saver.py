import tensorflow as tf
import time
import os
from FFBP.utilities.activation_functions import *

rw1 = tf.constant([[0.432171, 0.448781], [-0.038413, 0.036489]])
rw2 = tf.constant([[0.272080, 0.081714]])
rb1 = tf.constant([[-0.27659, -0.40250]])
rb2 = tf.constant([[0.27930]])

w1 = tf.Variable(rw1, dtype=tf.float32)
b1 = tf.Variable(rb1, dtype=tf.float32)
w2 = tf.Variable(rw2, dtype=tf.float32)
b2 = tf.Variable(rb2, dtype=tf.float32)

init = tf.initialize_all_variables()
saver = tf.train.Saver({'w_1': w1, 'b_1': b1, 'w_2':w2, 'b_2':b2})
with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, os.getcwd()+"/xor_params.ckpt")
    print("Model saved in file: {}".format(save_path))
