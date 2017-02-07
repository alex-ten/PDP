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


path_to_trainset = os.getcwd() + '/FFBP/data/embeddings_data.txt'

trainSet = DataSet(path_to_trainset)
testSet = DataSet(path_to_trainset)

# ----------------------------- BUILD -----------------------------

image = tf.placeholder(tf.float32, shape=[None,8], name='input')
label = tf.placeholder(tf.float32, shape=[None,8], name='target')

hidden = Layer(input = image,
                size = 5,
                act = actf.linear,
                layer_name = 'hidden1',
                layer_type = 'hidden')
hidden.init_wrange([-1,1,1])

output =  Layer(input = hidden,
                size = 8,
                act = actf.softmax,
                layer_name = 'output',
                layer_type = 'output')
output.init_wrange([-1,1,2])

w2v_model = model([image], [hidden, output], label)
w2v = Network(w2v_model, name='Word2Vec')

# ----------------------------- SETUP -----------------------------

w2v.train_set = trainSet
w2v.test_set = testSet
w2v.init_weights()

# Change these values to explore hyperparameters
w2v.config(loss = errf.cross_entropy,
           train_batch_size = 12,
           learning_rate = .02,
           momentum = 0,
           test_func = evalf.tce,
           permute = False,
           ecrit = 0.01)

w2v.tnt(3000,100)
w2v.test(True)
w2v.showerr()

code.interact(local = locals())
