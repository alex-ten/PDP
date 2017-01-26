import code
import os
import utilities.activation_functions as actf
import utilities.evaluation_functions as evalf
import tensorflow as tf
from utilities.model import model

import utilities.error_functions as errf
from FFBP.classes.DataSet import DataSet
from FFBP.classes.Layer import Layer
from FFBP.classes.Network import Network

path_to_trainset = os.getcwd()+'/FFBP/data/f_net838.txt'
trainSet = DataSet(path_to_trainset)
testSet = trainSet

input_layer = tf.placeholder(tf.float32, shape=[None,8], name='input')
targets = tf.placeholder(tf.float32, shape=[None,8], name='target')

hidden = Layer(input_tensor = input_layer,
	 	size = 3,
		act = actf.sigmoid,
		layer_name = 'hidden', 
		layer_type = 'hidden')
hidden.set_wrange([-1,1,1])


output = Layer(input_tensor = hidden.act, 
		size = 8, 
		act = actf.sigmoid,
		layer_name = 'output', 
		layer_type = 'output')
output.set_wrange([-1,1,1])

model838 = model(input_layer, [hidden, output], targets)
net838 = Network(model838, name='net838')

net838.train_set = trainSet
net838.test_set = testSet

net838.initconfig(loss = errf.squared_error, train_batch_size = 8, learning_rate = .3, momentum = .9, test_func = evalf.tss, test_scope ='all')

code.interact(local=locals())

# def main():
#
# 	net838.tnt(500,50,0)
# 	net838.test(vis=True)
#
# if __name__=='__main__': main()