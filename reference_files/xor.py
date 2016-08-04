import tensorflow as tf

# The code below implements the FFBP problem from PDP Handbook

batch_size = 1 # Batch size of 4 is equivalent epoch wise updating of network parameters
numepochs = 30
l_rate = 0.5
momentum = 0.9

inpats = [[0,0],
          [0,1],
          [1,0],
          [1,1]]
outpats = [[0],
           [1],
           [1],
           [0]]

rw1 = tf.constant([[0.432171, 0.448781], [-0.038413, 0.036489]])
rw2 = tf.constant([[0.272080, 0.081714]])
rb1 = tf.constant([[-0.27659, -0.40250]])
rb2 = tf.constant([[0.27930]])


# shape 2 and 1 in shape are numbers of units
inp_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 2), name='inslot')
target_placeholder = tf.placeholder(tf.float32, shape=(batch_size,1), name='targets')

with tf.name_scope('hidden'):
    w1 = tf.Variable(rw1, dtype=tf.float32, name='weight')
    b1 = tf.Variable(rb1, dtype=tf.float32, name='bias')
    netinp = tf.matmul(inp_placeholder, w1, transpose_b=True, name='netinp') + b1
    hidden = tf.nn.sigmoid(netinp)

with tf.name_scope('output'):
    w2 = tf.Variable(rw2, dtype=tf.float32, name='weight')
    b2 = tf.Variable(rb2, dtype=tf.float32, name='bias')
    netinp = tf.matmul(hidden, w2, transpose_b=True, name='netinp') + b2
    output = tf.nn.sigmoid(netinp)

tss = tf.reduce_sum(tf.squared_difference(target_placeholder, output))
pairs = {inp_placeholder: inpats, target_placeholder: outpats}
loss = tf.reduce_sum(tf.squared_difference(target_placeholder, output)/2)
train_step = tf.train.MomentumOptimizer(l_rate,momentum).minimize(loss)


step = False

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    i=0
    print('Before training:')
    print('HIDDEN:\n', sess.run(hidden, feed_dict=pairs))
    print('OUTPUTS:\n', sess.run(output, feed_dict=pairs))
    print('loss: ', sess.run(loss, feed_dict=pairs))
    print('tss: ', sess.run(tss, feed_dict=pairs))
    print('\nStarting training...')
    while i < numepochs:
        print('* EPOCH {}'.format(i+1))
        tf.scalar_summary(loss.op.name, loss)
        sess.run(train_step, feed_dict=pairs)
        print('tss: ', sess.run(tss, feed_dict=pairs))
        i+=1
        if i==numepochs: print('Training complete!\n')
    print('HIDDEN:\n', sess.run(hidden, feed_dict=pairs))
    print('OUTPUTS:\n',sess.run(output, feed_dict=pairs))
    print('LOSS:\n', sess.run(loss, feed_dict=pairs))
    print('TSS:\n', sess.run(tss, feed_dict=pairs))


