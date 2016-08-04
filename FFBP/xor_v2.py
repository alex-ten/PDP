import tensorflow as tf
from FFBP.classes.DataSet import load_data
from FFBP.classes.Layer import Layer
from FFBP.utilities.activation_functions import *
from FFBP.utilities.error_functions import *
from FFBP.utilities.run_training import *

DS = load_data('xor_files/f_XOR.txt')

batch_size = 4
lrate = 0.5
mrate = 0.9
num_epochs = 300

X = tf.placeholder(tf.float32, shape=[batch_size,2], name='input') # image / img_pat
target = tf.placeholder(tf.float32, shape=[batch_size,1], name='target') # label / lbl_pat

hidden1 = Layer(input_tensor = X,
                size = 2,
                wrange = [-1,1],
                act = sigmoid,
                layer_name = 'hidden1',
                seed = 3)

output =  Layer(input_tensor = hidden1.activations,
                size = 1,
                wrange = [-1,1],
                act = sigmoid,
                layer_name = 'output',
                seed = 3)

mymodel = (X, hidden1, output, target)

with tf.Session() as sess:
    run_training(model = mymodel,
                 dataset = DS,
                 num_epochs = num_epochs,
                 learning_rate = lrate,
                 momentum = mrate,
                 error = squared_error,
                 batch_size = batch_size,
                 evaluation = tss,
                 checkpoint = 100,
                 permute = False,
                 _restore_XOR = 'xor_files/xor_params.ckpt')