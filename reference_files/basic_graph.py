import tensorflow as tf
import numpy as np

# Operations with matrices as graph input flow
a = tf.placeholder(tf.float32, [2, 4])
b = tf.placeholder(tf.float32, [4, 2])

# Apply multiplication
mul = tf.matmul(a, b)

# Construct a `Session` to execute the graph
sess = tf.Session()

# Create matrices with defined dimensions and fill them with random values
rand_array_a = np.random.rand(2, 4)
rand_array_b = np.random.rand(4, 2)

# Execute the graph and print the resulting matrix
print(sess.run(mul, feed_dict={a: rand_array_a, b: rand_array_b}))

# Close the session
sess.close()