from FFBP.classes.DataSet import load_data
import tensorflow as tf
import time
import os

#-------------------------- The code below implements the FFBP problem from PDP Handbook ------------------------------

# Choose the update frequency (1 ~ pattern-wise learning, 4 ~ epoch-wise, 2 ~ update after each 2 training examples)
batch_size = 4
# Set the number of training epochs
numepochs = 5
# Select learning rate
l_rate = 0.5
# Select momentum rate
momentum = 0.9
# Choose the error measure, pass it to variable loss as string ("cross entropy" or "squared error")
error_measure = 'squared error'

# Create a DataSet object by loading a txt file. The txt file must be structured in a particular way (see file f_XOR.txt)
data_set = load_data('ex_XOR/f_XOR.txt')

# Create input and output placeholders
x_ = tf.placeholder(tf.float32, shape=(None, 2), name='inslot')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='targets')

# Initialize weights, biases and define the operations for the forward flow
with tf.name_scope('hidden'):
    w1 = tf.Variable(tf.random_uniform([2,2]), dtype=tf.float32, name='weights')
    b1 = tf.Variable(tf.random_uniform([1,2]), dtype=tf.float32, name='biases')
    netinp = tf.matmul(x_, w1, transpose_b=True, name='netinp') + b1
    hidden = tf.nn.sigmoid(netinp, name='activations')
with tf.name_scope('output'):
    w2 = tf.Variable(tf.random_uniform([1,2]), dtype=tf.float32, name='weights')
    b2 = tf.Variable(tf.random_uniform([1,1]), dtype=tf.float32, name='biases')
    netinp = tf.matmul(hidden, w2, transpose_b=True, name='netinp') + b2
    output = tf.nn.sigmoid(netinp, name='activations')

# Define the performance measure
tss = tf.reduce_sum(tf.squared_difference(y_, output))

# Define the error measure
if error_measure=='squared error':
    loss = tf.reduce_sum(tf.squared_difference(y_, output) / 2, name='squared_error')
elif error_measure == 'cross entropy':
    def clipped(x):
        # this handles cases when y * tf.log(y') outputs NaN
        return tf.clip_by_value(x,1e-10,1.0)
    loss = -tf.reduce_sum(y_ * tf.log(clipped(output)) + (1 - y_) * tf.log(clipped(1 - output)), name='cross_entropy')

# Define the optimization algorithm and feet the loss function to minimize
# train_step = tf.train.MomentumOptimizer(l_rate,momentum).minimize(loss) #TODO uncomment

opt = tf.train.MomentumOptimizer(l_rate, momentum)
dEds = opt.compute_gradients(loss)
# for pair in dEds: print(pair)
learn = opt.apply_gradients(dEds)

# Construct a a dictionary with values to feed into placeholders
pairs = {x_: data_set.images, y_: data_set.labels}

# Define the variable initialization operation
init_op = tf.initialize_all_variables()

# Define the saver operation to load pre-existing xor_files
# Keys in the dictionary must be the same ones that were used to store the xor_files (see saver.py file)
saver_op = tf.train.Saver({'w_1': w1, 'b_1': b1, 'w_2': w2, 'b_2': b2})

# Define summary operation(s)
loss_summary = tf.scalar_summary('loss', loss)
tss_summary = tf.scalar_summary('tss', tss) # loss and tss are equivalent
w1_summary = tf.histogram_summary(w1.name, w1) # This and the next 5 lines generate summaries for weights, biases and activaitons
b1_summary = tf.histogram_summary(b1.name, b1)
w2_summary = tf.histogram_summary(w2.name, w2)
b2_summary = tf.histogram_summary(b2.name, b2)
h_summary = tf.histogram_summary(hidden.name, hidden)
o_summary = tf.histogram_summary(output.name, output)

# Merge all summaries to make Summary Writing easier
merged = tf.merge_all_summaries()

# Append Summary protocol buffers to the event file at the current directory
summary_writer = tf.train.SummaryWriter(os.getcwd()+'/events', graph=tf.get_default_graph())



# Run the graph
sess = tf.Session()

sess.run(init_op)
saver_op.restore(sess, "ex_XOR/xor_params.ckpt")
start_time = time.time()
i=0
while i < numepochs:
    batch_xs, batch_ys = data_set.next_batch(batch_size)
    _, loss_val, TSS = sess.run([learn, loss, tss], feed_dict={x_: batch_xs, y_: batch_ys})
    summary_str = sess.run(merged, feed_dict={x_: batch_xs, y_: batch_ys})
    summary_writer.add_summary(summary_str, i)
    summary_writer.flush()
    i+=1
    if i % 1 == 0:
        print('Step {}: error = {} | tss = {}'.format(i, loss_val, TSS))
    if i==numepochs:
        print('Training complete!\n')
        end_time = time.time()
        print('Training took about {} second(s):'.format(round(end_time-start_time,3)))
        print('Final tss = {}'.format(sess.run(tss, feed_dict={x_: batch_xs, y_: batch_ys})))

# To visualize the graph and tensor values with TensorBoard, run the tensorboard.py file with the
# --logdir='path' option, where path is the location of the event files created
# during training. File tensorboard.py should be in the tensorboard folder in the
# tensorflow directory. For example, in the terminal type:
#   $ python3.5 /tensorflow/tensorboard/tensorboard.py --logdir='FFBP/events'
# If everything is right, you should see the following output in the terminal:
#           Starting TensorBoard  on port 6006
#           (You can navigate to http://0.0.0.0:6006)
# Then, open a web browser and go to the address "http://0.0.0.0:6006"


