import code
import tensorflow as tf
import utilities.activation_functions as actf
import utilities.evaluation_functions as evalf
import utilities.error_functions as errf
from utilities.model import model
from classes.DataSet import DataSet
from classes.Layer import Layer
from classes.Network import Network

# ----------------------------- BUILD -----------------------------
try:
    ET = DataSet('exercises/ex_EightThings/f_EightThings.txt')
except FileNotFoundError:
    ET = DataSet('ex_EightThings/f_EightThings.txt')

item = tf.placeholder(tf.float32, shape=[None,8], name='item')
relation = tf.placeholder(tf.float32, shape=[None,4], name='relation')
labels = tf.placeholder(tf.float32, shape=[None,36], name='labels')

representation = Layer(
    input_tensor=item,
    size=8,
    act=actf.sigmoid,
    layer_name='representation',
    layer_type='hidden')
representation.set_wrange([-.45, .45, 1])

hidden = Layer(
    # concatenate representation.activations and relation (name properly for better visualization)
    input_tensor=tf.concat(1,[representation.act, relation], name='representation/relation'),
    size=12,
    act=actf.sigmoid,
    layer_name='hidden',
    layer_type='hidden')
hidden.set_wrange([-.45, .45, 1])

attribute = Layer(
    input_tensor=hidden.act,
    size=36,
    act=actf.sigmoid,
    layer_name='attribute',
    layer_type='output')
attribute.set_wrange([-.45, .45, 1])

eight_things = model([item, relation], [representation, hidden, attribute], labels)
et_net = Network(eight_things, name = '8t_network')

# ----------------------------- SETUP -----------------------------

et_net.train_set = ET
et_net.test_set = ET
et_net.configure(train_batch_size = 32,
                 learning_rate = 0.1,
                 momentum = 0,
                 permute = True,
                 ecrit = 2.5,
                 loss = errf.squared_error,
                 test_func = evalf.tss,
                 test_scope='all',
                 )
et_net.init_weights()

# code.interact(local=locals())

# ------------------------------- RUN ------------------------------

def main():
    et_net.test(vis=True)
    # for i in range(10):
    #     if et_net._terminate: continue
    #     et_net.test(vis=True)
    #     et_net.train(200, vis=True, ckpt_freq=False)

if __name__=="__main__":  main()