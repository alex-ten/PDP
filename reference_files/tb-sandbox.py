import tensorflow as tf
import numpy as np
import os

a = tf.Variable(0, dtype=tf.float32)
b = tf.random_uniform([0,0], minval=-10, maxval=10, dtype=tf.float32, seed=None, name=None)

i = 0
tf.scalar_summary(a.op.name, a)
summary_op = tf.merge_all_summaries()

with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter(os.getcwd(), sess.graph)
    sess.run(tf.initialize_all_variables())
    while i < 100:
        a = a + b
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str)
        summary_writer.flush()