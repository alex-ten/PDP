# ----------------------------- PRELIMINARIES -----------------------------

import os
import code
import tensorflow as tf
import utilities.activation_functions as actf
import utilities.evaluation_functions as evalf
import utilities.error_functions as errf
from utilities.model import model
from FFBP.classes.DataSet import DataSet
from FFBP.classes.Layer import Layer
from FFBP.classes.Network import Network

path = os.path.expanduser('~')+'/PDP/FFBP/data/f_net838.txt'
trainSet = DataSet(path)
testSet = trainSet

# ----------------------------- CONSTRUCTION -------------------------------

PHinp = tf.placeholder(tf.float32, shape=[None, 8], name='input')
PHout = tf.placeholder(tf.float32, shape=[None, 8], name='target')

hid = Layer(input = PHinp,
			size = 3,
			act = actf.sigmoid,
			layer_name = 'hidden',
			layer_type = 'hidden')
hid.init_wrange([-1,1,1])

output = Layer(input = hid,
		size = 8, 
		act = actf.sigmoid,
		layer_name = 'output', 
		layer_type = 'output')
output.init_wrange([-1,1,1])


model838 = model(PHinp, [hid, output], PHout)
net838 = Network(model838, name='net838')

net838.train_set = trainSet
net838.test_set = testSet

net838.initconfig(loss = errf.squared_error,
				  train_batch_size = 8,
				  learning_rate = .3,
				  momentum = .9,
				  permute = False,
				  ecrit = 0.01,
				  test_func = evalf.tss)


# --------------------------------- RUN -----------------------------------

code.interact(local=locals())