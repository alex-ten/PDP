# CPR
import tensorflow as tf
import numpy as np

def extract(s):
    s = s.split('/', 1)[0]
    s = s.split(':', 1)[0]
    return s

def orthogonal_initializer(scale = 1.1):
    # From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    def _initializer(shape, dtype=tf.float32):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape) #this needs to be corrected to float32
        #print('you have initialized one orthogonal matrix.')
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer


class Layer(object):
    def __init__(self, input_tensor, size, init, act, layer_name, layer_type='nd', bias=True, bias_val=None):
        self.input_tensor = input_tensor
        self.sender_size = int(input_tensor.get_shape()[1])
        self.sender_name = extract(input_tensor.name)
        self.size = size
        self.layer_name = layer_name
        self.layer_type = layer_type
        self.bias_on = bias
        self.bias_val = bias_val
        self.init = init
        self.actf = act
        if not callable(init):
            if len(init) == 3: seed = init[-1]
            else: seed = None
            self.init_wrange(seed)
        else: self.init_orthogonal()


    def init_orthogonal(self):
        with tf.variable_scope(self.layer_name, reuse=False):
            self.W = tf.get_variable('weights', [self.sender_size, self.size], tf.float32, initializer = self.init)
            self.net = tf.matmul(self.input_tensor, self.W, name = 'net')
            if self.bias_on:
                self.b = tf.get_variable('biases', [self.size], tf.float32)
                if self.bias_val != None:
                    self.b = self.b.assign(tf.constant(self.bias_val, shape = [self.size]))
                self.net = self.net + self.b
            if self.actf != None:
                self.act = self.actf(self.net)

    def init_wrange(self, rand_seed):
        if self.init == 0:
            init_W = tf.zeros((self.size, self.sender_size))
            init_b = tf.zeros((1, self.size))
        else:
            init_W = tf.random_uniform((self.size, self.sender_size),
                                       minval=self.init[0],
                                       maxval=self.init[1],
                                       seed=rand_seed)
            init_b = tf.random_uniform((1, self.size),
                                       minval=self.init[0],
                                       maxval=self.init[1],
                                       seed=rand_seed + 1)
        with tf.name_scope('weights'):
            self.W = tf.Variable(init_W, dtype=tf.float32)
        with tf.name_scope('biases'):
            self.b = tf.Variable(init_b, dtype=tf.float32)
        with tf.name_scope('net'):
            self.net = tf.matmul(self.input_tensor, self.W, transpose_b=True) + self.b
        with tf.name_scope('act'):
            self.act = self.actf(self.net)
        with tf.name_scope('input'):
            self.inp = tf.add(self.input_tensor, 0)

    def __str__(self):
        return '<Layer object>'


class RecurrentLayer(object):
    def __init__(self, input_tensor, size, wrange, act, layer_name, seed=None, layer_type='nd'):
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
                init_cW = tf.zeros((self.size, self.sender_size+self.size))
            else:
                init_W = tf.random_uniform((self.size, self.sender_size), minval=self.wrange[0],
                                           maxval=self.wrange[1], seed=self.rand_seed)
                init_b = tf.random_uniform((1, self.size), minval=self.wrange[0], maxval=self.wrange[1],
                                           seed=self.rand_seed + 1)
                init_cW = tf.random_uniform((self.size, self.sender_size + self.size), minval=self.wrange[0],
                                           maxval=self.wrange[1], seed=self.rand_seed)
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
        self.layer_type = layer_type
        self.context = tf.placeholder(tf.float32, [None, self.size], name = self.layer_name + ' context')

    def __str__(self):
        return '<Layer object>'

    def loop_on(self):
        current_input_tensor = self.input_tensor
        context = self.context
        self.input_tensor = tf.concat(1, [current_input_tensor, context], name = '{}_with_context'.format(self.input_tensor.name)) #todo check if this gives correct name
