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

def wrange_initializer(wrange):
    if wrange == 0:
        return tf.constant_initializer(0)
    if len(wrange) == 3:
        seed = wrange[-1]
    else:
        seed = None
    return tf.random_uniform_initializer(minval = wrange[0],
                                         maxval = wrange[1],
                                         seed = seed)


class Layer(object):
    def __init__(self, input_tensor, size, act, layer_name, layer_type='nd', bias=True, bias_val=None, stop_grad=False):
        self.input_tensor = input_tensor
        self.sender_size = int(input_tensor.get_shape()[1])
        self.sender_name = extract(input_tensor.name)
        self.size = size
        self.layer_name = layer_name
        self.layer_type = layer_type
        self.bias_on = bias
        self.bias_val = bias_val
        self.actf = act
        self.stop_grad = stop_grad
        self.W = self.b = None

    def set_orthogonal(self, scope=1.1):
        with tf.variable_scope(self.layer_name, reuse=False):
            self.W = tf.get_variable(name='weights',
                                     shape=[self.sender_size, self.size],
                                     dtype=tf.float32,
                                     initializer=orthogonal_initializer(scope))
            self.net = tf.matmul(self.input_tensor, self.W, name = 'net')
            if self.bias_on:
                self.b = tf.get_variable('biases', [self.size], tf.float32)
                if self.bias_val != None:
                    self.b = self.b.assign(tf.constant(self.bias_val, shape = [self.size]))
                self.net = self.net + self.b
            if self.actf != None:
                self.act = self.actf(self.net)

    def set_wrange(self, wrange=0):
        with tf.variable_scope(self.layer_name, reuse=False, initializer = wrange_initializer(wrange)):
            self.W = tf.get_variable(name = 'weights',
                                     shape = [self.sender_size, self.size],
                                     dtype = tf.float32)
            self.b = tf.get_variable(name = 'biases',
                                     shape = [1, self.size],
                                     dtype = tf.float32)
        with tf.name_scope('net'):
            self.net = tf.matmul(self.input_tensor, self.W) + self.b
            if self.stop_grad: self.net = tf.stop_gradient(self.net)
        with tf.name_scope('act'):
            self.act = self.actf(self.net)

    def __str__(self):
        return '<Layer object>'


class RecurrentLayer(object):
    def __init__(self, input_tensor, size, init_state, act, layer_name, layer_type='nd', bias=True, bias_val=None,
                 stop_grad=False):
        self.input_tensor = input_tensor
        self.sender_size = int(input_tensor.get_shape()[1])
        self.sender_name = extract(input_tensor.name)
        self.size = size
        self.layer_name = layer_name
        self.layer_type = layer_type
        self.bias_on = bias
        self.bias_val = bias_val
        self.actf = act
        self.stop_grad = stop_grad
        self.W = self.rW = self.b = None
        self.init_state = self.curr_state = init_state
        self.inp = []

    def update_state(self):
        self.curr_state = self.act

    def flush_state(self):
        self.curr_state = self.init_state

    def set_orthogonal(self, scope=1.1):
        with tf.variable_scope(self.layer_name, reuse=False):
            self.W = tf.get_variable(name = 'weights',
                                     shape = [self.sender_size, self.size],
                                     dtype = tf.float32,
                                     initializer = orthogonal_initializer(scope))
            self.rW = tf.get_variable(name = 'recurrent_weights',
                                      shape = [self.size, self.size],
                                      dtype = tf.float32,
                                      initializer = orthogonal_initializer((scope)))
            self.inp.append(tf.matmul(self.input_tensor, self.W, name='inp_curr'))
            self.inp.append(tf.matmul(self.curr_state, self.rW, name='inp_prev'))
            if self.stop_grad: self.inp[1] = tf.stop_gradient(self.inp[1])
            self.net = self.inp[0] + self.inp[1]
            if self.bias_on:
                self.b = tf.get_variable('biases', [self.size], tf.float32)
                if self.bias_val != None:
                    self.b = self.b.assign(tf.constant(self.bias_val, shape=[self.size]))
                self.net += self.b
            if self.actf != None:
                self.act = self.actf(self.net)

    def set_wrange(self, wrange=0):
        with tf.variable_scope(self.layer_name, reuse=False, initializer = wrange_initializer(wrange)):
            self.W = tf.get_variable(name = 'weights',
                                     shape = [self.sender_size, self.size],
                                     dtype = tf.float32)
            self.rW = tf.get_variable('recurrent_weights',
                                      [self.size, self.size],
                                      tf.float32)
            self.b = tf.get_variable(name = 'biases',
                                     shape = [1, self.size],
                                     dtype = tf.float32)
        with tf.name_scope('net'):
            self.inp.append(tf.matmul(self.input_tensor, self.W))
            self.inp.append(tf.matmul(self.curr_state, self.rW))
            if self.stop_grad: self.inp[1] = tf.stop_gradient(self.inp[1])
            self.net = self.inp[0] + self.inp[1] + self.b
        with tf.name_scope('act'):
            self.act = self.actf(self.net)


    def __str__(self):
        return '<RecurrentLayer object>'