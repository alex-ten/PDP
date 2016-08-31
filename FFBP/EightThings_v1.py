import tensorflow as tf
import FFBP.utilities.evaluation_functions as evalf
import FFBP.utilities.activation_functions as actf
import FFBP.utilities.error_functions as errf
from FFBP.classes.DataSet import load_data
from FFBP.classes.Layer import Layer
from FFBP.classes.Network import Network
from FFBP.utilities.model import model

# ----------------------------- BUILD -----------------------------

ET = load_data('ex_EightThings/f_EightThings.txt')

item = tf.placeholder(tf.float32, shape=[None,8], name='item')
relation = tf.placeholder(tf.float32, shape=[None,4], name='relation')
labels = tf.placeholder(tf.float32, shape=[None,36], name='labels')

representation = Layer(
    input_tensor=item,
    size=8,
    wrange=[-.45, .45],
    act=actf.sigmoid,
    layer_name='representation',
    seed=1)

hidden = Layer(
    input_tensor=tf.concat(1,[representation.activations, relation]), # concatenate representation.activations and relation'
    size=12,
    wrange=[-.45, .45],
    act=actf.sigmoid,
    layer_name='hidden',
    seed=2)

attribute = Layer(
    input_tensor=hidden.activations,
    size=36,
    wrange=[-.45, .45],
    act=actf.sigmoid,
    layer_name='attribute',
    seed=3)

eight_things = model([item, relation], [representation, hidden, attribute], labels)
et_model = Network(eight_things)

# ----------------------------- SETUP -----------------------------

et_model.configure(train_batch_size = 8,
                   test_batch_size = 8,
                   learning_rate = 0.1,
                   momentum = 0,
                   permute = True,
                   ecrit = 2.5,
                   loss = errf.squared_error,
                   test_func = evalf.tss,
                   test_scope='all',
                   )
et_model.init_weights()

# ------------------------------- RUN ------------------------------

et_model.tnt(2000, ET, ET, 200, 2000)
# et_model.interact(train_set=ET, test_set=ET)
et_model.visualize_loss()