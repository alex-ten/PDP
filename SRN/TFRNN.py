import tensorflow as tf
from SRN.FSM.DataSet import DataSet
from FFBP.utilities.printProgress import printProgress
import numpy as np
from matplotlib import style
style.use('ggplot')
import matplotlib.pyplot as plt
from constructors.Layer import wrange_initializer


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
train_targ = tf.placeholder(tf.int32, shape = [batch_size, seq_len], name ='target')
rnnCell = tf.nn.rnn_cell.BasicRNNCell(hid_size, data_dim, tf.nn.tanh)

soft_W = tf.get_variable('softmax_W', [hid_size, data_dim], initializer=wrange_initializer([-0.5, 0.5, 2]))
soft_b = tf.get_variable('softmax_b', [data_dim], initializer=wrange_initializer([-0.5, 0.5, 2]))

def fprop(inp, compute_loss = True):
    batch_size = inp.get_shape()[0]
    num_steps = inp.get_shape()[1]
    hid_states = []
    state = tf.zeros([batch_size, hid_size], dtype = tf.float32)
    for t in range(num_steps):  # sets up the copies and their feeding relationships
        # the next line is the critical line that ties the weights across the copies
        if t > 0:
            tf.get_variable_scope().reuse_variables()
        # the next line sets up the feed from one copy to the next
        hid_act, newstate = rnnCell(inp[:, t, :], state)
        # use next line for SRN, use the second line for BPTT for num_steps
        if not BPTT:
            state = tf.stop_gradient(newstate) #SRN
        else:
            state = newstate #BPTT
        hid_states.append(hid_act)
    hid_states = tf.reshape(tf.concat(1, hid_states), [-1, hid_size])
    logits = tf.matmul(hid_states, soft_W) + soft_b
    if not compute_loss:
        return tf.nn.softmax(logits)
    else:
        labels = tf.reshape(train_targ, [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
        # labels = tf.reshape(tf.concat(1, __targ__), [-1, data_dim])
        # labels = tf.reshape(__targ, [-1, data_dim])
        # loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
        return loss

loss = fprop(inp = train_inp, compute_loss = True)
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
    sess.run(tf.initialize_all_variables())

    test_batch_size = 4
    test_inp = tf.placeholder(tf.float32, shape=[test_batch_size, seq_len, data_dim], name='item')

    plot_data_mean = []
    for epoch in range(num_epochs):
        printProgress(epoch, num_epochs, prefix='{}'.format(epoch), barLength=20)
        batch_xs, batch_ys, __ = data.next_batch(batch_size, ind_batch_Y = True)
        _, cost = sess.run([adam_step, loss], feed_dict = {train_inp: batch_xs, train_targ: batch_ys})
        average_loss = np.sum(cost) / seq_len
        plot_data_mean.append(average_loss)
        if epoch % test_step == 0 or epoch == num_epochs - 1:
            print('\nTest before epoch {}'.format(epoch))
            print('mean sequence loss = {}'.format(average_loss))
            simple_test(test_inp, test_data, test_batch_size, p=True)

    vis = input('visualize? [y/n] >> ')
    if len(plot_data_mean) > 0 and vis=='y':
        plt.plot(plot_data_mean, lw = 2, color = 'green')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Training')
        plt.show()

if __name__=='__main__': main()