import code
import tensorflow as tf
import FFBP.utilities.evaluation_functions as evalf
import FFBP.utilities.activation_functions as actf
import FFBP.utilities.error_functions as errf
from FFBP.constructors.DataSet import DataSet
from FFBP.constructors.Network import Network
from FFBP.constructors.Layer import Layer
from FFBP.utilities.model import model

trainSet = DataSet('f_net838.txt')
testSet = trainSet

input_layer = tf.placeholder(tf.float32, shape=[None,8], name='input')
targets = tf.placeholder(tf.float32, shape=[None,8], name='target')

hidden = Layer(input_tensor = input_layer,
	 	size = 3, 
	 	wrange = [-1,1], 
		act = actf.sigmoid, 
		layer_name = 'hidden', 
		seed = 1, 
		layer_type = 'hidden')

output = Layer(input_tensor = hidden.act, 
		size = 8, 
		wrange = [-1,1], 
		act = actf.sigmoid, 
		layer_name = 'output', 
		seed = 1,
		layer_type = 'output')

model838 = model(input_layer, [hidden, output], targets)
net838 = Network(model838, name='net838')

net838.train_set = trainSet
net838.test_set = testSet

net838.init_and_configure(loss = errf.squared_error, train_batch_size = 8, learning_rate = .3, momentum = .9, test_func = evalf.tss, test_scope = 'all')

net838.test()
net838.train(50)
net838.tnt(500,50,0)
net838.train(100,1)
net838.test(1)

code.interact(local=locals())
 
