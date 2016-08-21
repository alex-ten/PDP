import numpy

import tensorflow as tf
import FFBP.utilities.evaluation_functions as evalf
import FFBP.utilities.activation_functions as actf
import FFBP.utilities.error_functions as errf
from FFBP.classes.DataSet import load_data
from FFBP.classes.Network import Network
from FFBP.classes.Layer import Layer
from FFBP.utilities.model import model

trainSet = load_data('ex_XOR/f_XOR.txt')
testSet = load_data('ex_XOR/f_XOR.txt')

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
mynet = Network(xor, name='XOR Network')

# ----------------------------- SETUP -----------------------------

batch_size = 4
lrate = 0.5
mrate = 0.9
num_epochs = 300
ecrit = 0.01
error = errf.squared_error

mynet.init_weights()
mynet.restore('ex_XOR/xor_params.ckpt')
mynet.configure(loss = error,
                batch_size = batch_size,
                learning_rate = lrate,
                momentum = mrate,
                test_func = evalf.tss,
                test_scope = 'all')

# --------------------------- INTERACT -----------------------------

mynet.tnt(300, train_set = trainSet, test_set = testSet, train_batch_size =  4, test_batch_size = 4, snp_checkpoint=50)
# mynet.interact(DS, ts)
# mynet.visualize_loss()
mynet.off()