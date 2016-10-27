import tensorflow as tf
import numpy as np
import time

from constructors.RNNCell import RNNCell
from constructors.Layer import wrange_initializer
from SRN.FSM.DataSet import DataSet
from utilities import mask
from utilities.printProgress import printProgress
from utilities.model import model


class RecurrentNetwork():
    def __init__(self, model, bptt=True):
        self.inp =
        self.targ = targ
        self.cell = rnn_cell
        self.bptt = bptt
        self.train_set = None
        self.test_set = None
        self.sess = tf.InteractiveSession()
        self.graph = tf.get_default_graph()
        self._settings = {}

    def init_and_configure(self,
                           loss,
                           train_batch_size,
                           learning_rate=0.5,
                           momentum=0.9,
                           permute=False,
                           ecrit=0.01,
                           test_func=None,
                           test_scope='all'):

        self.init_weights()
        self.configure(loss, train_batch_size, learning_rate,
                       momentum, permute, ecrit, test_func, test_scope)

    def init_weights(self):
        # Initialize weights and biases
        Wb_vars = self.graph.get_collection('Wb')
        init_Wb_vars = tf.initialize_variables(Wb_vars)
        self.sess.run(init_Wb_vars)

    def configure(self,
                  loss,
                  train_batch_size,
                  learning_rate=0.5,
                  momentum=0.9,
                  permute=False,
                  ecrit=0.01,
                  test_func=None,
                  test_scope='all'):
        self._loss = loss(self.model['labels'], self.model['network'][-1].act)
        self._opt = tf.train.MomentumOptimizer(learning_rate, momentum)
        self._settings['loss_func'] = loss
        self._settings['train_batch'] = train_batch_size
        self._settings['lrate'] = learning_rate
        self._settings['mrate'] = momentum
        self._settings['permute'] = permute
        self._settings['ecrit'] = ecrit
        self._settings['test_func'] = test_func
        self._settings['scope'] = test_scope
        self._settings['opt_task'] = self._opt.minimize(self._loss)
        self._settings['saver'] = tf.train.Saver()
        for l in self.model['network']:
            # When run in current session tf.gradients returns a list of numpy arrays with
            # batch_size number of rows and Layer.size number of columns.
            # That is, the rows of the returned arrays contain partial derivatives of loss with respect
            # to the argument tensor (here the loss tensor) of each unit in the layer given a particular input
            l.ded_net = tf.gradients(self._loss, l.net)
            l.ded_act = tf.gradients(self._loss, l.act)
            l.ded_W = tf.gradients(self._loss, l.W)
            l.ded_b = tf.gradients(self._loss, l.b)
        hyper_parameters = [('Learning rate:', self._settings['lrate']),
                            ('Momentum rate:', self._settings['mrate']),
                            ('Error:', self._settings['loss_func']),
                            ('Batch size:', self._settings['train_batch']),
                            ('Permuted mode:', self._settings['permute']),
                            ('Ecrit:', self._settings['ecrit'])]
        store(collections.OrderedDict(hyper_parameters), self.logpath)
        init = init_rest()
        self.sess.run(init)

    def fprop(self, inp, targ, use_mask=None, compute_loss=True):
        batch_size = inp.get_shape()[0]
        num_steps = inp.get_shape()[1]
        hid_states = []
        targ_list = []
        inp_list = []
        cell.set_init_state(tf.zeros([batch_size, hid_size], dtype=tf.float32))
        with tf.variable_scope("RNN"):  # this sets the scope for the ‘reuse_variables’ below
            for t in range(num_steps):  # sets up the copies and their feeding relationships
                # the next line is the critical line that ties the weights across the copies
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                # the next line sets up the feed from one copy to the next
                newstate = cell.step(inp[:, t, :])
                inp_list.append(inp[:, t, :])
                targ_list.append(targ[:, t, :])
                # use next line for SRN, use the second line for BPTT for num_steps
                if not BPTT:
                    state = tf.stop_gradient(newstate)  # SRN
                else:
                    state = newstate  # BPTT
                hid_states.append(state)
        hid_states = tf.reshape(tf.concat(1, hid_states), [-1, hid_size])
        logits = tf.matmul(hid_states, W) + b
        if not compute_loss:
            return tf.nn.sigmoid(logits)
            # return tf.nn.softmax(logits)
        else:
            # ------------- SPARSE SOFTMAX CROSS ENTROPY WITH LOGITS ----------------
            # labels = tf.reshape(targ, [-1])
            # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)

            # ---------------- SOFTMAX CROSS ENTROPY WITH LOGITS --------------------
            # labels = tf.reshape(tf.concat(1, targ_list), [-1, data_dim])
            # loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels)

            # ---------------- SIGMOID CROSS ENTROPY WITH LOGITS --------------------
            labels = tf.reshape(tf.concat(1, targ_list), [-1, data_dim])
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels)

            # --------------------- MAYBE RETURN VARIABLES ---------------------------
            # inps = tf.reshape(tf.concat(1, inp_list), [-1, data_dim])
            # predictions = tf.nn.softmax(logits)
            # loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(predictions), 1))
            if use_mask is not None:
                _mask = use_mask
                return mask.mask(x=loss,
                                 seq_lengths=_mask,
                                 max_len=int(num_steps),
                                 batch_size=int(batch_size))
            return loss

    def _train(self, num_epochs):
        pass

    def _test(self):
        pass



def demo():
    data = DataSet('pickles/240x50.pkl')
    data.raw2onehot()
    data.raw2inds()
    test_data = data

    learning_rate = 0.1
    batch_size = 10
    hid_size = 3
    data_dim = len(data.unique) # The number of unique tokens in the entire set = length of one-hot vectors
    seq_len = num_steps = data.max_length - 1
    hid_actf = tf.nn.sigmoid

    BPTT = True

    train_inp = tf.placeholder(tf.float32, shape = [batch_size, seq_len, data_dim], name ='item')
    train_targ_vec = tf.placeholder(tf.float32, shape = [batch_size, seq_len, data_dim], name ='target')
    train_targ_ind = tf.placeholder(tf.int32, shape = [batch_size, seq_len], name ='target')
    lengths_placeholder = tf.placeholder(tf.int32, shape = [batch_size])

    cell = RNNCell(data_dim, hid_size, hid_actf, 'RNN')

    RN_model = model(train_inp, cell, train_targ_vec)

    W = tf.get_variable('Weights', [hid_size, data_dim], initializer=wrange_initializer([-0.5, 0.5, 2]))
    b = tf.get_variable('biases', [data_dim], initializer=wrange_initializer([-0.5, 0.5, 2]))

    RNN = RecurrentNetwork(train_inp, train_targ_vec, cell)


if __name__=='__main__': demo()