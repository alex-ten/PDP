import numpy as np
import tensorflow as tf

import utilities.activation_functions as actf
from FFBP.classes import Layer

batch_size = 1
data_dim = 4
hid_size = 3
learning_rate = 1e-3
num_steps = 2

inp_vec = np.array([[0,0,0,1],
                    [1,0,0,0]])

targ_vec1 = inp_vec
targ_vec2 = np.array([[0,0,1,0],
                      [0,1,0,0]])

_input = tf.placeholder(tf.float32,shape=[None,data_dim])
_target = tf.placeholder(tf.float32,shape=[None,data_dim])

hid = RecurrentLayer(input_tensor=_input,
                     size=hid_size,
                     init_state = tf.zeros((1,hid_size)),
                     act = actf.sigmoid,
                     layer_name = 'hidden',
                     layer_type = 'recurrent',
                     stop_grad = True)
hid.init_orthogonal()

out = Layer(input_tensor = hid.act,
            size = data_dim,
            act = actf.sigmoid,
            layer_name = 'prediction',
            layer_type = 'output')
out.init_weights([-1, 1])

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

outputs = []
with tf.variable_scope(hid.layer_name):  # this sets the scope for the ‘reuse_variables’ below
    hid.flush_state()
    for t in range(num_steps):  # sets up the copies and their feeding relationships
        # the next line is the critical line that ties the weights across the copies
        if t > 0: tf.get_variable_scope().reuse_variables()
        # the next line sets up the feed from one copy to the next
        output = hid.act.eval(feed_dict={_input:np.reshape(inp_vec[t], (1,len(inp_vec[t])))})
        hid.update_state()
        outputs.append(output)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.constant(np.array(outputs),tf.float32,(2,4)) - _target), 1))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

for output in enumerate(outputs):
    sess.run(train_step, feed_dict={_target:targ_vec1})

for o in outputs:
    print(o)