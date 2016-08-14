import tensorflow as tf
import FFBP.utilities.evaluation_functions as evalf
import FFBP.utilities.activation_functions as actf
import FFBP.utilities.error_functions as errf
import FFBP.utilities.train as train
from FFBP.classes.DataSet import load_data
from FFBP.classes.Network import Network
from FFBP.classes.Layer import Layer
from FFBP.utilities.model import model

DS = load_data('ex_XOR/f_XOR.txt')

image = tf.placeholder(tf.float32, shape=[None,2], name='input') # image / img_pat
label = tf.placeholder(tf.float32, shape=[None,1], name='target') # label / lbl_pat

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

mynet = Network(xor)
mynet.dataset = DS
mynet.init_weights()
mynet.restore('ex_XOR/xor_params.ckpt')

batch_size = 4
lrate = 0.5
mrate = 0.9
num_epochs = 300
ecrit = 0.01
error = errf.squared_error

while mynet.ecrit_not_reached:
    mynet.train(num_epochs = None,
                learning_rate = lrate,
                momentum = mrate,
                loss = error,
                batch_size = batch_size,
                checkpoint = 300,
                permute = False)
    mynet.test(batch_size = batch_size,
               eval = evalf.tss,
               loss = error)

mynet.print_logdir()
mynet.off()