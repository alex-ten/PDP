# Import TensorFlow.
import tensorflow as tf

# Basic operations with some constants

# Create constants a and b

# Build a data flow graph
a = tf.constant(10, dtype=tf.float32) # tf.constant is an op which returns tensor a=10
b = tf.constant(5, dtype=tf.float32)

# Create a session to evaluate the symbolic expression
sess = tf.Session()

# Trigger an evaluation of the data flow graph.
print("Here's the graph constants:")
print('a = {}'.format(sess.run(a)))
print('b = {}'.format(sess.run(b)))
print("Some of things we can do with them:")
print('a + 17 = {}'.format(sess.run(a+17)))
print("a + b = {}".format(sess.run(a + b)))
print("a * b = {}".format(sess.run(a * b)))
print("a / b = {}".format(sess.run(a * b)))
print("sqrt(a-1) - b = {}".format(sess.run(tf.sqrt(a-1)-b)))

# Close the session
sess.close()