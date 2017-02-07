import os
import code
import tensorflow as tf

import utilities.activation_functions as actf
import utilities.evaluation_functions as evalf
from utilities.model import model
import utilities.error_functions as errf
from FFBP.classes.DataSet import DataSet
from FFBP.classes.Layer import Layer
from FFBP.classes.Network import Network


path_to_trainset = os.getcwd() + '/FFBP/data/f_XOR.txt'
path_to_params = os.getcwd() + '/FFBP/data/xor_params1.ckpt'

trainSet = DataSet(path_to_trainset)
testSet = DataSet(path_to_trainset)

# ----------------------------- BUILD -----------------------------

image = tf.placeholder(tf.float32, shape=[None,2], name='input')
label = tf.placeholder(tf.float32, shape=[None,1], name='target')

hidden1 = Layer(input = image,
                size = 2,
                act = actf.sigmoid,
                layer_name = 'hidden1',
                layer_type = 'hidden')

output =  Layer(input = hidden1,
                size = 1,
                act = actf.sigmoid,
                layer_name = 'output',
                layer_type = 'output')

xor_model = model([image], [hidden1, output], label)
xor = Network(xor_model, name='XOR Network')

# ----------------------------- SETUP -----------------------------

xor.train_set = trainSet
xor.test_set = testSet
xor.init_weights()

# Change these values to explore hyperparameters
xor.config(loss = errf.squared_error,
           train_batch_size = 4,
           learning_rate = .25,
           momentum = 0.9,
           test_func = evalf.tss,
           permute = False,
           ecrit = 0.01,
           wrange=[-0.5,0.5])

xor.restore(path_to_params) # <-- Comment this line out for random weights

xor.tnt(3000,30)
xor.test(True)

code.interact(local = locals())
