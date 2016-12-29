import os
print(os.getcwd())
import utilities.activation_functions as actf
import utilities.evaluation_functions as evalf
from utilities.model import model
import utilities.error_functions as errf
from classes.DataSet import DataSet
from classes.Layer import Layer
from classes.Network import Network
# import sys
# for i in sys.path: print(i)

import tensorflow as tf

path_to_trainset = os.getcwd()+'/FFBP/exercises/ex_XOR/f_XOR.txt'
path_to_params = os.getcwd()+'/FFBP/exercises/ex_XOR/xor_params1.ckpt'

trainSet = DataSet(path_to_trainset)
testSet = trainSet

# ----------------------------- BUILD -----------------------------

image = tf.placeholder(tf.float32, shape=[None,2], name='input')
label = tf.placeholder(tf.float32, shape=[None,1], name='target')

hidden1 = Layer(input_tensor = image,
                size = 2,
                act = actf.sigmoid,
                layer_name = 'hidden1',
                layer_type = 'hidden',
                stop_grad=False)
hidden1.set_wrange([-1,1,2])


output =  Layer(input_tensor = hidden1.act,
                size = 1,
                act = actf.sigmoid,
                layer_name = 'output',
                layer_type = 'output')
output.set_wrange([-1,1,1])

xor_model = model([image], [hidden1, output], label)
xor_net = Network(xor_model, name='XOR Network')

# ----------------------------- SETUP -----------------------------

xor_net.train_set = trainSet
xor_net.test_set = testSet

xor_net.init_weights()
xor_net.restore(path_to_params)
xor_net.configure(loss = errf.squared_error,
                  train_batch_size = 4,
                  learning_rate = 0.5,
                  momentum = 0.9,
                  test_func = evalf.tss,
                  test_scope = 'all')

# code.interact(local = locals())

def demo():
    xor_net.tnt(330,30,0)
    xor_net.test(vis=True)

if __name__=="__main__":  demo()

