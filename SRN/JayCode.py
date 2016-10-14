import tensorflow as tf
from SRN.FSM.DataSet import DataSet
from constructors.Layer import orthogonal_initializer

data = DataSet('pickles/240.pkl')
data.raw2onehot()


learning_rate = 1e-3
num_epochs = 2

batch_size = 4
hid_size= 3
data_dim = len(data.unique) # The number of unique characters in the entire set = length of one-hot vectors
num_steps = data.max_length

BPTT = False

item = tf.placeholder(tf.float32, shape = [batch_size, data_dim], name = 'item')
targ = tf.placeholder(tf.float32, shape= [batch_size, data_dim], name = 'target')
context = tf.placeholder(tf.float32, shape = [batch_size, hid_size], name = 'context')

with tf.variable_scope("RNN"):
    W = tf.get_variable(name = 'weights',
                        shape = [data_dim + hid_size, hid_size],
                        dtype = tf.float32,
                        initializer = orthogonal_initializer())
    b = tf.get_variable(name = 'biases',
                        shape = [1, hid_size],
                        dtype = tf.float32,
                        initializer = orthogonal_initializer())

initial_state = state = tf.zeros([batch_size, hid_size], dtype=tf.float32)
hid_states = []

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for epoch in range(num_epochs):
    batch_xs, _ = data.next_batch(batch_size)
    with tf.variable_scope("RNN"):  #this sets the scope for the ‘reuse_variables’ below
        for t in range(num_steps):  #sets up the copies and their feeding relationships
            #The next line is the critical line that ties the weights across the copies
            if t > 0:
                tf.get_variable_scope().reuse_variables()
            #the next line sets up the feed from one copy to the next
            newstate = activation = tf.sigmoid(tf.matmul(tf.concat(1, [batch_xs[:, t, :], state]), W)) + b
            # use next line for SRN, use the second line for BPTT for num_steps
            if not BPTT:
                state = tf.stop_gradient(newstate) #SRN
            else:
                state = newstate #BPTT
            hid_states.append(activation)



