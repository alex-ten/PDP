import tensorflow as tf
import FFBP.utilities.evaluation_functions as evalf
import FFBP.utilities.activation_functions as af
import FFBP.utilities.error_functions as ef
import FFBP.utilities.train as train
from FFBP.classes.DataSet import load_data
from FFBP.classes.Layer import Layer
from FFBP.utilities.model import model

ET = load_data('ex_EightThings/f_EightThings.txt')

batch_size = 8
lrate = 0.05
mrate = 0.9
num_epochs = 10000

item = tf.placeholder(tf.float32, shape=[batch_size,8], name='item')
relation = tf.placeholder(tf.float32, shape=[batch_size,4], name='relation')
labels = tf.placeholder(tf.float32, shape=[batch_size,36], name='labels')

representation = Layer(
    input_tensor=item,
    size=8,
    wrange=[-1, 1],
    act=af.sigmoid,
    layer_name='representation',
    seed=1)

hidden = Layer(
    input_tensor=tf.concat(1,[representation.activations, relation]), # concatenate representation.activations and relation'
    size=12,
    wrange=[-1, 1],
    act=af.sigmoid,
    layer_name='hidden',
    seed=2)

attribute = Layer(
    input_tensor=hidden.activations,
    size=36,
    wrange=[-1, 1],
    act=af.sigmoid,
    layer_name='attribute',
    seed=3)

eight_things = model([item, relation] ,[representation, hidden, attribute], labels)

with tf.Session() as sess:
    train.SGD(model = eight_things,
              dataset = ET,
              num_epochs = num_epochs,
              learning_rate = lrate,
              momentum = mrate,
              error = ef.cross_entropy,
              batch_size = batch_size,
              evaluation = evalf.tss,
              checkpoint = 300,
              permute = False,
              _restore_XOR = False)