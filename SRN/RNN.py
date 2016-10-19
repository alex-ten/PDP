import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

from SRN.FSM.DataSet import DataSet
from constructors.RNNCell import RNNCell
from constructors.Layer import wrange_initializer

data = DataSet('pickles/a_to_x.pkl')
data.raw2onehot()
data.raw2inds()
test_data = data

learning_rate = 0.1
batch_size = 2
hid_size = 3
data_dim = len(data.unique) # The number of unique characters in the entire set = length of one-hot vectors
seq_len = num_steps = data.max_length - 1
hid_actf = tf.nn.sigmoid

class RNN():
    def __init__(self, inp, targ, rnn_cell, bptt=True):
        self.inp = inp
        self.targ = targ
        self.cell = rnn_cell
        self.bptt = bptt
        self.train_data = None
        self.test_data = None
        self.sess = tf.InteractiveSession()
        self.graph = tf.get_default_graph()

BPTT = True

train_inp = tf.placeholder(tf.float32, shape = [batch_size, seq_len, data_dim], name ='item')
train_targ_vec = tf.placeholder(tf.float32, shape = [batch_size, seq_len, data_dim], name ='target')
train_targ_ind = tf.placeholder(tf.int32, shape = [batch_size, seq_len], name ='target')

cell = RNNCell(data_dim, hid_size, hid_actf, 'RNN')

W = tf.get_variable('Weights', [hid_size, data_dim], initializer=wrange_initializer([-0.5, 0.5, 2]))
b = tf.get_variable('biases', [data_dim], initializer=wrange_initializer([-0.5, 0.5, 2]))

def fprop(inp, targ, compute_loss = True):
    batch_size = inp.get_shape()[0]
    num_steps = inp.get_shape()[1]
    hid_states = []
    targ_list = []
    inp_list = []
    cell.set_init_state(tf.zeros([batch_size, hid_size], dtype = tf.float32))
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
                state = tf.stop_gradient(newstate) # SRN
            else:
                state = newstate # BPTT
            hid_states.append(state)
    hid_states = tf.reshape(tf.concat(1, hid_states), [-1, hid_size])
    logits = tf.matmul(hid_states, W) + b
    if not compute_loss:
        return tf.nn.softmax(logits)
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
        inps = tf.reshape(tf.concat(1, inp_list), [-1, data_dim])
        predictions = tf.nn.softmax(logits)
        # loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(predictions), 1))
        return loss

loss  = fprop(inp = train_inp, targ = train_targ_vec, compute_loss = True)
sgd_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
mom_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)
adam_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

def simple_test(test_inp, dataset, test_batch_size):
    np.set_printoptions(precision = 2, suppress = True)
    x, y = dataset.next_batch(test_batch_size)
    if batch_size == 1:
        print("Testing sequence {}: ".format(dataset.raw[dataset._batch_ind-1]))
    else:
        print("Testing sequences {}: ".format(
            dataset.raw[dataset._batch_ind - test_batch_size: dataset._batch_ind]))
    out = fprop(inp = test_inp, targ=train_targ_vec, compute_loss = False)
    np_out = out.eval(feed_dict = {test_inp: x})
    print("Test inputs: ")
    print(np.reshape(np.array(x), (-1, data_dim)))
    print("Target outputs: ")
    print(np.reshape(np.array(y), (-1, data_dim)))
    print("Test outputs: ")
    print(np_out)


def main():
    num_epochs = 5000
    test_step = 500
    test_batch_size = 2
    test_inp = tf.placeholder(tf.float32, shape=[test_batch_size, seq_len, data_dim], name='item')

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    simple_test(test_inp, test_data, test_batch_size)

    plot_data_mean = []
    global_start = time.time()
    for epoch in range(num_epochs):
        start = time.time()
        batch_xs, batch_ys = data.next_batch(batch_size)
        _, cost = sess.run([adam_step, loss], feed_dict = {train_inp: batch_xs, train_targ_vec: batch_ys})
        seq_loss = np.sum(cost) / seq_len
        plot_data_mean.append(seq_loss)
        if epoch % test_step == 0 or epoch == num_epochs - 1:
            print('epoch {}:'.format(epoch))
            print('  mean sequence loss: {}'.format(round(seq_loss, 4)))
            print('  time: {} s'.format(round(time.time() - start, 4)))
    print('Total time: {}'.format(round(time.time()- global_start, 4)))

    simple_test(test_inp, test_data, test_batch_size)

    print('Entering interactive test mode...')
    int_test = None
    while int_test != 'quit':
        if int_test == None:
            cell.set_init_state(tf.zeros([1, hid_size], dtype=tf.float32))
            int_test = input('Enter B to begin a new sequence (or q to quit): ')
            if int_test == 'q': break
        inp_vec = tf.cast(data.get_oh(int_test), tf.float32)
        print('{}:    {}'.format(int_test, inp_vec.eval()))
        hid_act = cell.step(tf.reshape(inp_vec, [1, data_dim]))
        logits = tf.matmul(hid_act, W) + b
        out = tf.nn.softmax(logits).eval()
        np.set_printoptions(precision=2, suppress=True)
        print('OUT: {}'.format(np.squeeze(out)))
        min_max = np.argsort(np.squeeze(out))
        first = min_max[-1]
        second = min_max[-2]
        _map = list(data.oh_map.keys())
        opt1 = _map[first]
        opt2 = _map[second]
        if opt1 == 'E':
            print('Network predicts E (end of sequence)'.format(opt1, opt2))
            act = input('One more? [y/n] > ')
            if act == 'n': break
            else:
                int_test = None
        else:
            print('Network predicts {} or {}'.format(opt1, opt2))
            int_test = input('Continue? [{}/{}/quit] > '.format(opt1, opt2))


    vis = input('visualize? [y/n] >> ')
    if len(plot_data_mean) > 0 and vis=='y':
        xax = list(range(0, num_epochs, int(test_step)))
        xax.append(num_epochs - 1)
        plt.plot(plot_data_mean, lw = 2, color = 'green')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Training')
        plt.show()

if __name__=='__main__': main()