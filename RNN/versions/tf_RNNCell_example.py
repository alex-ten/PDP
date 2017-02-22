import tensorflow as tf

batch_size = 1
hid_size = 3

cell = tf.nn.rnn_cell.BasicRNNCell(num_units = hid_size, activation = tf.nn.sigmoid)

init_state = cell.zero_state(batch_size, tf.float32)

inp_vec = tf.random_uniform((batch_size, 5))

with tf.variable_scope('RNN'):
    out, state = cell(inp_vec, init_state)

sigm = tf.nn.sigmoid(inp_vec)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

rnn_variables = [v for v in tf.global_variables()
                 if v.name.startswith('RNN')]

# Input array
print('x = {}; type: {}'.format(inp_vec.eval(), type(inp_vec.eval())))

# Sigmoid activation of input array (no bias, W = identity matrix)
print('f(x) = {}, type = {}'.format(sigm.eval(), type(sigm.eval())))

print('cell_init = {}'.format(init_state.eval()))

print('cell_out_size = {}'.format(cell.output_size))

print('cell_output = {}'.format(out.eval()))

print('cell_state () = {}'.format(state.eval()))

print('RNN variables:\nvar1 = {}\nvar2 = {}'.format(rnn_variables[0].eval(),rnn_variables[1].eval()))

