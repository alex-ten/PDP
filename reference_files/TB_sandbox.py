import tensorflow as tf
import os
import random
import time


cwd = os.getcwd()

# y = ax + b

a = tf.constant(3, dtype=tf.float32)
b = tf.constant(5, dtype=tf.float32)
x_ = tf.placeholder(dtype=tf.float32, shape=None)

y = a * x_ + b

init = tf.initialize_all_variables()
summary_op = tf.histogram_summary('output', y)
summary_writer = tf.train.SummaryWriter(cwd, graph=tf.get_default_graph())

with tf.Session() as sess:
    i=0
    sess.run(init)
    while i < 10:
        a = random.random() * 10
        print(sess.run(y, feed_dict={x_: a}))
        q = sess.run(summary_op, feed_dict={x_: a})
        summary_writer.add_summary(q, i)
        summary_writer.flush()
        time.sleep(0.3)
        i+=1
