from importlib import reload
import tensorflow as tf
import FFBP.utilities.evaluation_functions as evalf
import FFBP.utilities.activation_functions as actf
import FFBP.utilities.error_functions as errf
from FFBP.constructors.DataSet import DataSet
from FFBP.constructors.Network import Network
from FFBP.constructors.Layer import Layer
from FFBP.utilities.model import model

trainSet = DataSet('exercises/ex_XOR/f_XOR.txt')
testSet = trainSet

# ----------------------------- BUILD -----------------------------

image = tf.placeholder(tf.float32, shape=[None,2], name='input')
label = tf.placeholder(tf.float32, shape=[None,1], name='target')

hidden1 = Layer(input_tensor = image,
                size = 2,
                wrange = [-1,1],
                act = actf.sigmoid,
                layer_name = 'hidden1',
                seed = 9,
                layer_type = 'hidden')

output =  Layer(input_tensor = hidden1.act,
                size = 1,
                wrange = [-1,1],
                act = actf.sigmoid,
                layer_name = 'output',
                seed = 2,
                layer_type = 'output')

xor_model = model([image], [hidden1, output], label)
xor_net = Network(xor_model, name='XOR Network')

# ----------------------------- SETUP -----------------------------

xor_net.train_set = trainSet
xor_net.test_set = testSet

xor_net.init_weights()
xor_net.restore('exercises/ex_XOR/xor_params.ckpt')
xor_net.configure(loss = errf.squared_error,
                  train_batch_size = 4,
                  test_batch_size = 4,
                  learning_rate = 0.5,
                  momentum = 0.9,
                  test_func = evalf.tss,
                  test_scope = 'all')

def demo():
    for i in range(3):
        xor_net.test(vis=True)
        input('hit me')
        xor_net.train(30, vis=True, ckpt_freq=False)

if __name__=="__main__":  demo()

