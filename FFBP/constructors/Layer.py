# CPR
import tensorflow as tf

def extract(s):
    s = s.split('/', 1)[0]
    s = s.split(':', 1)[0]
    return s

class Layer(object):
    def __init__(self, input_tensor, size, wrange, act, layer_name, seed=None):
        self.input_tensor = input_tensor
        self.sender_size = int(input_tensor.get_shape()[1])
        self.sender_name = extract(input_tensor.name)
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
            with tf.name_scope('biases'):
                self.b = tf.Variable(init_b, dtype=tf.float32, collections=['Wb'])
            with tf.name_scope('net'):
                self.net = tf.matmul(input_tensor, self.W, transpose_b=True) + self.b
            with tf.name_scope('act'):
                self.act = act(self.net)
            with tf.name_scope('input'):
                self.inp = tf.add(input_tensor, 0)

    def __str__(self):
        return '<Layer object>'