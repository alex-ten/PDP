import tensorflow as tf
from RNN.classes.RNNModels import Basic_LSTM_Model, Basic_RNN_Model

class Configs(object):
    init_scale = 0.1
    learning_rate = 0.05
    max_grad_norm = 5
    num_layers = 1
    num_steps = 3
    hidden_size = 10
    max_epoch = 10
    max_max_epoch = 10
    keep_prob = 1
    lr_decay = 1
    batch_size = 4
    vocab_size = 8

batch_size = 4
num_steps = 3

config = Configs()

inp = [1,2,3,0,1,6,3,0,4,2,5,0,4,7,5,0]
inp = tf.convert_to_tensor(inp, name="raw_data", dtype=tf.int32)

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(2, state_is_tuple=True)

cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

initial_state = cell.zero_state(batch_size, tf.float32)

print(type(initial_state[0]))
print(initial_state[0])


with tf.Session() as sess:
    state = sess.run(initial_state)

# for j in range(2):
feed_dict = {}
# print(state)
try:
    for (c, h) in enumerate(initial_state):
        # print('i:{}\n(c:{}\n,h):{}'.format(0, c, h))
        # feed zero values to the model's initial state if state is a tuple
        feed_dict[c] = state[0].c + 1
        feed_dict[h] = state[0].h + 5
        for k,v in feed_dict.items(): print(k,v)
except TypeError:
    feed_dict[initial_state] = state + 5

with tf.Session() as sess:
    act = tf.nn.sigmoid(initial_state)
    print(sess.run(initial_state, feed_dict))



