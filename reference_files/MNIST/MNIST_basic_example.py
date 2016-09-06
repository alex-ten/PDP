from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

#------------------------ Defining tensords ---------------------------------------

x = tf.placeholder(tf.float32, [None, 784])     # Input data slot todo look up tf.placeholder()

W = tf.Variable(tf.zeros([784, 10]))            # Weights todo look up tf.Variable()

b = tf.Variable(tf.zeros([10]))                 # Biases

#------------------------ Adding ops (operations) -----------------------------------------

y = tf.nn.softmax(tf.matmul(x, W) + b)          # Feed-forward todo look up tf.nn and tf.nn.Softmax()

y_ = tf.placeholder(tf.float32, [None, 10])     # Output data slot

cross_entropy = -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]) # Define error op
# todo lookup tf.reduce_mean and tf.reduce_sum()

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)           # Choose optimizer
# todo lookup tf.train and tf.train.GradientDescentOptimizer and *args

init = tf.initialize_all_variables()    # Initialize all variables todo lookup tf.Session and variable initialization
sess = tf.Session()
sess.run(init)

# Each step of the loop, we get a "batch" of one hundred random data points from our training set.
# We run train_step feeding in the batches data to replace the placeholders.

for i in range(1000):
    if i % 100 == 0: print('Epoch: {}'.format(i))
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# ------------------------ Testing the model -----------------------------------------------
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # Gives a list of booleans where bool_i = y_i == y__i

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # Convert the list of booleans to accuracy percentage (True/True+False)

print(sess.run(accuracy, feed_dict={x: mnist._test.images, y_: mnist._test.labels}))