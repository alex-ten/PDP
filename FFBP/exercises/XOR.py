import tensorflow as tf
from constructors.DataSet import DataSet
from constructors.Network import Network

import FFBP.utilities.activation_functions as actf
import FFBP.utilities.error_functions as errf
import FFBP.utilities.evaluation_functions as evalf
from FFBP.utilities.model import model
from constructors.Layer import Layer

trainSet = DataSet('exercises/ex_XOR/f_XOR.txt')
testSet = trainSet

# ----------------------------- BUILD -----------------------------

image = tf.placeholder(tf.float32, shape=[None,2], name='input')
label = tf.placeholder(tf.float32, shape=[None,1], name='target')

hidden1 = Layer(input_tensor = image,
                size = 2,
                act = actf.sigmoid,
                layer_name = 'hidden1',
                layer_type = 'hidden',
                stop_grad=True)
hidden1.set_wrange([-1,1,2])


output =  Layer(input_tensor = hidden1.act,
                size = 1,
                act = actf.sigmoid,
                layer_name = 'output',
                layer_type = 'output')
output.set_wrange([-1,1,2])

xor_model = model([image], [hidden1, output], label)
xor_net = Network(xor_model, name='XOR Network')

# ----------------------------- SETUP -----------------------------

xor_net.train_set = trainSet
xor_net.test_set = testSet

xor_net.init_weights()
xor_net.restore('exercises/ex_XOR/xor_params.ckpt')
xor_net.configure(loss = errf.squared_error,
                  train_batch_size = 4,
                  learning_rate = 0.5,
                  momentum = 0.9,
                  test_func = evalf.tss,
                  test_scope = 'all')

# code.interact(local = locals())

def demo():
    #xor_net.tnt(300,30,0)
    #xor_net.test(vis=True)
    print(xor_net.model['network'][0].W.eval())
    xor_net.train(300,0,0)
    print(xor_net.model['network'][0].W.eval())

if __name__=="__main__":  demo()

