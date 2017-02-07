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

# ----------------------------- BUILD -----------------------------
path_to_trainset = os.getcwd()+'/FFBP/data/f_EightThings.txt'
ET = DataSet(path_to_trainset)

item = tf.placeholder(tf.float32, shape=[None,8], name='item')
relation = tf.placeholder(tf.float32, shape=[None,4], name='relation')
labels = tf.placeholder(tf.float32, shape=[None,36], name='labels')

representation = Layer(
    input=item,
    size=8,
    act=actf.sigmoid,
    layer_name='representation',
    layer_type='hidden')
representation.init_wrange([-.45, .45, 1])

hidden = Layer(
    input = [representation, relation],
    size=12,
    act=actf.sigmoid,
    layer_name='hidden',
    layer_type='hidden')
hidden.init_wrange([-.45, .45, 1])

attribute = Layer(
    input=hidden,
    size=36,
    act=actf.sigmoid,
    layer_name='attribute',
    layer_type='output')
attribute.init_wrange([-.45, .45, 1])

eight_things = model([item, relation], [representation, hidden, attribute], labels)
et_net = Network(eight_things, name = '8t_network')

# ----------------------------- SETUP -----------------------------
et_net.train_set = ET
et_net.test_set = ET
et_net.config(train_batch_size = 32,
              learning_rate = 0.1,
              momentum = 0,
              permute = True,
              ecrit = 2.5,
              loss = errf.squared_error,
              test_func = evalf.tss
              )
et_net.init_weights()
# et_net.test(vis=True)
code.interact(local=locals())

# ------------------------------- RUN ------------------------------
#
# def main():
#     et_net.tnt(330,30,0)
#     et_net.test(vis=True)
#
# if __name__=="__main__":  main()