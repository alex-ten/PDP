import numpy

import tensorflow as tf
import FFBP.utilities.evaluation_functions as evalf
import FFBP.utilities.activation_functions as actf
import FFBP.utilities.error_functions as errf
import FFBP.utilities.train as train
from FFBP.classes.DataSet import load_data
from FFBP.classes.Network import Network
from FFBP.classes.Layer import Layer
from FFBP.utilities.model import model

DS = load_data('ex_XOR/f_XOR.txt')

# ----------------------------- BUILD -----------------------------

image = tf.placeholder(tf.float32, shape=[None,2], name='input')
label = tf.placeholder(tf.float32, shape=[None,1], name='target')

hidden1 = Layer(input_tensor = image,
                size = 2,
                wrange = [-1,1],
                act = actf.sigmoid,
                layer_name = 'hidden1',
                seed = 9)

output =  Layer(input_tensor = hidden1.activations,
                size = 1,
                wrange = [-1,1],
                act = actf.sigmoid,
                layer_name = 'output',
                seed = 2)

xor = model([image], [hidden1, output], label)
mynet = Network(xor)

# ----------------------------- SETUP -----------------------------

batch_size = 4
lrate = 0.5
mrate = 0.9
num_epochs = 300
ecrit = 0.01
error = errf.squared_error

mynet.dataset = DS
mynet.init_weights()
mynet.restore('ex_XOR/xor_params.ckpt')
mynet.configure(learning_rate = lrate,
                momentum = mrate,
                loss = error)

# ----------------------------- TRAIN ------------------------------
while mynet._below_ecrit:
    mynet.test(batch_size = batch_size,
               eval = evalf.tss,
               loss = error)
    mynet.train(num_epochs = None,
                batch_size = batch_size,
                ecrit = 0.01,
                checkpoint = 300,
                permute = False)
mynet.visualize()
mynet.print_logdir()
mynet.off()