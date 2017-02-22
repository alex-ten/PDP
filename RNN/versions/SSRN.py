from RNN.versions.ops import *

hid_dim = 20
learning_rate = 1e-3
data_dim = 3
mb_dim = 32
seq_len = 10
_input = tf.placeholder(tf.float32,shape=[None,data_dim])
_prev_hid = tf.placeholder(tf.float32,shape=[None,hid_dim])
_target = tf.placeholder(tf.float32,shape=[None,data_dim])

hid = linear(tf.concat(axis=1,values=[_input,_prev_hid]),hid_dim,'hid',tf.nn.relu)
out = linear(hid,data_dim,'out')

loss = tf.reduce_mean(tf.reduce_sum(tf.square(out-_target),1))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#fake data
data_x = np.random.rand(mb_dim,seq_len+1,data_dim)
cum_loss = 0

for i in range(1000):
    last_hid = np.zeros((mb_dim, hid_dim))
    for j in range(seq_len):
        _,last_hid,cur_loss = sess.run([train_step,hid,loss],feed_dict={_input:data_x[:,j],_prev_hid:last_hid,_target:data_x[:,j+1]})
    cum_loss += cur_loss
    if i % 100 == 0:
        print(i,cum_loss)
        cum_loss = 0


