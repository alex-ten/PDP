import tensorflow as tf
from constructors.Layer import orthogonal_initializer, wrange_initializer

initializer = wrange_initializer([-1,1,1])

class RNNCell(object):
    def __init__(self, inp_size, size, actf, name='RNN'):
        self.state = None
        self.nonlin = actf
        with tf.variable_scope(name):
            self.W = tf.get_variable(name='weights',
                                     shape=[inp_size + size, size],
                                     dtype=tf.float32,
                                     initializer=initializer)
            self.b = tf.get_variable(name='biases',
                                     shape=[1, size],
                                     dtype=tf.float32,
                                     initializer=initializer)

    def set_init_state(self, state):
        self.state = state

    def step(self, x):
        # update the hidden state
        self.state = self.nonlin(tf.matmul(tf.concat(1, [x, self.state]), self.W)) + self.b
        return self.state

