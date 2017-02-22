import numpy as np
import tensorflow as tf
from matplotlib import style

from RNN.FSM.DataSet import DataSet
from utilities.printProgress import printProgress
from utilities import mask

style.use('ggplot')
import matplotlib.pyplot as plt
from FFBP.classes.Layer import wrange_initializer


data = DataSet('pickles/240x50.pkl')
data.raw2onehot()
data.raw2inds()
test_data = data
for k,v in data.oh_map.items():
    print(k,v)

learning_rate = 0.1
batch_size = 10
hid_size = 3
data_dim = len(data.unique) # The number of unique characters in the entire set = length of one-hot vectors
seq_len = num_steps = data.max_length - 1
hid_actf = tf.nn.sigmoid

BPTT = False

train_inp = tf.placeholder(tf.float32, shape = [batch_size, seq_len, data_dim], name ='item')
train_targ = tf.placeholder(tf.float32, shape = [batch_size, seq_len, data_dim], name ='target')
rnnCell = tf.nn.rnn_cell.BasicRNNCell(hid_size, None, tf.nn.tanh)

soft_W = tf.get_variable('softmax_W', [hid_size, data_dim], initializer=wrange_initializer([-0.5, 0.5, 2]))
soft_b = tf.get_variable('softmax_b', [data_dim], initializer=wrange_initializer([-0.5, 0.5, 2]))

def fprop(inp, targ, use_mask = None, compute_loss = True):
    batch_size = inp.get_shape()[0]
    num_steps = inp.get_shape()[1]
    hid_states = []
    targ_list = []
    inp_list = []
    state = tf.zeros([batch_size, hid_size], dtype = tf.float32)
    for t in range(num_steps):  # sets up the copies and their feeding relationships
        # the next line is the critical line that ties the weights across the copies
        if t > 0:
            tf.get_variable_scope().reuse_variables()
        # the next line sets up the feed from one copy to the next
        hid_act, newstate = rnnCell(inp[:, t, :], state)
        targ_list.append(targ[:, t, :])
        # use next line for SRN, use the second line for BPTT for num_steps
        if not BPTT:
            state = tf.stop_gradient(newstate) #SRN
        else:
            state = newstate #BPTT
        hid_states.append(hid_act)
    hid_states = tf.reshape(tf.concat(axis=1, values=hid_states), [-1, hid_size])
    logits = tf.matmul(hid_states, soft_W) + soft_b
    if not compute_loss:
        return tf.nn.sigmoid(logits), hid_states
        # return tf.nn.softmax(logits)
    else:
        # ------------- SPARSE SOFTMAX CROSS ENTROPY WITH LOGITS ----------------
        # labels = tf.reshape(targ, [-1])
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)

        # ---------------- SOFTMAX CROSS ENTROPY WITH LOGITS --------------------
        # labels = tf.reshape(tf.concat(1, targ_list), [-1, data_dim])
        # loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels)

        # ---------------- SIGMOID CROSS ENTROPY WITH LOGITS --------------------
        labels = tf.reshape(tf.concat(axis=1, values=targ_list), [-1, data_dim])
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        predictions = tf.nn.sigmoid(logits)

        # --------------------- MAYBE RETURN VARIABLES ---------------------------
        # inps = tf.reshape(tf.concat(1, inp_list), [-1, data_dim])
        # predictions = tf.nn.softmax(logits)
        # loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(predictions), 1))
        if use_mask is not None:
            _mask = use_mask
            masked_loss = mask.mask(x=loss,
                                    seq_lengths=_mask,
                                    max_len=int(num_steps),
                                    batch_size=int(batch_size))
            return masked_loss, predictions, hid_states
        return loss, predictions, hid_states

loss, predictions, hid_states = fprop(inp = train_inp, targ=train_targ, use_mask=None, compute_loss = True)

sgd_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
mom_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)
adam_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

def simple_test(test_inp, dataset, test_batch_size, p=False):
    np.set_printoptions(precision = 2, suppress = True)
    x, y, _ = dataset.next_batch(test_batch_size)
    _y = []
    for i in range(num_steps):
        _y.append(y[:,i,:])
    if p: print(x[:, 0, :])
    if batch_size == 1:
        if p: print("Testing sequence {}: ".format(dataset.raw[dataset._batch_ind-1]))
    else:
        if p: print("Testing sequences {}: ".format(
            dataset.raw[dataset._batch_ind - test_batch_size: dataset._batch_ind]))
    if p:
        print("Target outputs: ")
        print(np.reshape(np.array(_y), (-1, len(dataset.unique))))
    out = fprop(inp = test_inp, compute_loss = False)
    np_out = out.eval(feed_dict = {test_inp: x})
    if p:
        print("Test outputs: ")
        print(np_out)

def main():
    num_epochs = 1000
    test_step = 1000
    xax = list(range(0,num_epochs,int(test_step)))
    xax.append(num_epochs-1)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    test_batch_size = 4
    test_inp = tf.placeholder(tf.float32, shape=[test_batch_size, seq_len, data_dim], name='item')

    plot_data_mean = []
    for epoch in range(num_epochs):
        printProgress(epoch, num_epochs, prefix='{}'.format(epoch), barLength=20)
        batch_xs, batch_ys, __ = data.next_batch(batch_size)
        _, cost = sess.run([adam_step, loss], feed_dict = {train_inp: batch_xs, train_targ: batch_ys})
        average_loss = np.sum(cost) / seq_len
        plot_data_mean.append(average_loss)
        if epoch % test_step == 0 or epoch == num_epochs - 1:
            print('\nTest before epoch {}'.format(epoch))
            print('mean sequence loss = {}'.format(average_loss))
            # simple_test(test_inp, test_data, test_batch_size, p=True)

    vis = input('visualize? [y/n] >> ')
    if len(plot_data_mean) > 0 and vis=='y':
        plt.plot(plot_data_mean, lw = 2, color = 'green')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Training')
        plt.show()

if __name__=='__main__': main()