from importlib import reload
import tensorflow as tf
import FFBP.utilities.evaluation_functions as evalf
import FFBP.utilities.activation_functions as actf
import FFBP.utilities.error_functions as errf
from FFBP.constructors.DataSet import DataSet
from FFBP.constructors.Layer import Layer
from FFBP.constructors.Network import Network
from FFBP.utilities.model import model



# ----------------------------- BUILD -----------------------------

ET = DataSet('exercises/ex_EightThings/f_EightThings.txt')

item = tf.placeholder(tf.float32, shape=[None,8], name='item')
relation = tf.placeholder(tf.float32, shape=[None,4], name='relation')
labels = tf.placeholder(tf.float32, shape=[None,36], name='labels')

representation = Layer(
    input_tensor=item,
    size=8,
    wrange=[-.45, .45],
    act=actf.sigmoid,
    layer_name='representation',
    seed=1,
    layer_type='hidden')

hidden = Layer(
    # concatenate representation.activations and relation (name properly for neat visualization)
    input_tensor=tf.concat(1,[representation.act, relation], name='representation/relation'),
    size=12,
    wrange=[-.45, .45],
    act=actf.sigmoid,
    layer_name='hidden',
    seed=2,
    layer_type='hidden')

attribute = Layer(
    input_tensor=hidden.act,
    size=36,
    wrange=[-.45, .45],
    act=actf.sigmoid,
    layer_name='attribute',
    seed=3,
    layer_type='output')

eight_things = model([item, relation], [representation, hidden, attribute], labels)
et_net = Network(eight_things, name = '8t_network')

# ----------------------------- SETUP -----------------------------

et_net.train_set = ET
et_net.test_set = ET
et_net.configure(train_batch_size = 32,
                 test_batch_size = 32,
                 learning_rate = 0.1,
                 momentum = 0,
                 permute = True,
                 ecrit = 2.5,
                 loss = errf.squared_error,
                 test_func = evalf.tss,
                 test_scope='all',
                 )
et_net.init_weights()

# ------------------------------- RUN ------------------------------

def main():
    for i in range(10):
        if et_net._terminate: continue
        et_net.test()
        et_net.train(200, vis=True, ckpt_freq=False)

if __name__=="__main__":  main()