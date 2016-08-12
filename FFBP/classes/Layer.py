# CPR
import tensorflow as tf

class Layer(object):
    def __init__(self, input_tensor, size, wrange, act, layer_name, seed=None):
        self.input_tensor = input_tensor
        self.sender_size = int(input_tensor.get_shape()[1])
        self.size = size
        self.layer_name = layer_name
        self.wrange = wrange
        self.rand_seed = seed
        with tf.name_scope(layer_name):
            if wrange == 0:
                init_W = tf.zeros((self.size, self.sender_size))
                init_b = tf.zeros((1, self.size))
            else:
                init_W = tf.random_uniform((self.size, self.sender_size), minval=self.wrange[0], maxval=self.wrange[1], seed=self.rand_seed)
                init_b = tf.random_uniform((1, self.size), minval=self.wrange[0], maxval=self.wrange[1], seed=self.rand_seed+1)
            with tf.name_scope('weights'):
                self.W = tf.Variable(init_W, dtype=tf.float32, collections=['Wb'])
                self.variable_summaries(self.W, layer_name + '/weights')
            with tf.name_scope('biases'):
                self.b = tf.Variable(init_b, dtype=tf.float32, collections=['Wb'])
                self.variable_summaries(self.b, layer_name + '/biases')
            with tf.name_scope('netinp'):
                self.netinp = tf.matmul(input_tensor, self.W, transpose_b=True) + self.b
                tf.histogram_summary(layer_name + '/pre_activations', self.netinp)
            self.activations = act(self.netinp, 'activation')
            tf.histogram_summary(layer_name + '/activations', self.activations)

    def __str__(self):
        return '<Layer object>'

    def variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.scalar_summary('sttdev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)
