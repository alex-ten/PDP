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
testSet = trainSet

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

mynet.init_weights()
mynet.restore('ex_XOR/xor_params.ckpt')
mynet.configure(loss = errf.squared_error,
                train_batch_size = 4,
                test_batch_size = 4,
                learning_rate = 0.5,
                momentum = 0.9,
                test_func = evalf.tss,
                test_scope = 'all')

# --------------------------- INTERACT -----------------------------

# mynet.tnt(330, train_set = trainSet, test_set = testSet, train_batch_size =  4, test_batch_size = 4, snp_checkpoint=30)
mynet.interact(train_set=trainSet, test_set=testSet)
mynet.visualize_loss()
mynet.off()