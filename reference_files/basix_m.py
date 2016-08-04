# Import TensorFlow.
import tensorflow as tf

# Perform the same operations on constant matrices

# Build a dataflow graph
a = tf.constant([[2.0, 2.0], [5.0, 4.0]])
b = tf.constant([[1.0, 2.0], [1.0, 2.0]])
e = tf.matmul(a, b)

# Construct a `Session` to execute the graph
sess = tf.Session()

# Execute the graph and print the resulting matrix
print("Here's the graph constants:")
print('a = {}'.format(sess.run(a)))
print('b = {}'.format(sess.run(b)))
print("Some of things we can do with them:")
print('a * b = {}'.format(sess.run(e)))
print("a + b = {}".format(sess.run(a + b)))
print("a .* b = {}".format(sess.run(a * b)))
print("a / b = {}".format(sess.run(a * b)))
print("sqrt(a-1) - b =\n{}".format(sess.run(tf.sqrt(a-1)-b)))

# Close the session
sess.close()