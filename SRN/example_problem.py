import code
import tensorflow as tf
import FFBP.utilities.evaluation_functions as evalf
import FFBP.utilities.activation_functions as actf
import FFBP.utilities.error_functions as errf
from SRN.DataSet import DataSet
from FFBP.constructors.Network import Network
from FFBP.constructors.Layer import Layer
from FFBP.utilities.model import model

data = DataSet('240.pkl')
data.raw2onehot()

item = tf.placeholder(tf.float32, shape = [None, len(data.unique)], name = 'item')
context = tf.placeholder(tf.float32, shape = [None, 3], name = 'context')

inp = tf.concat(1, [item,context], name = 'item:context')

hidden = Layer(input_tensor = item,
               size = 3,
               wrange = [-1,1],
               act = actf.sigmoid,
               layer_name = 'hidden',
               seed = 1,
               layer_type = 'recurrent')

output = Layer(input_tensor = hidden.act,
               size = len(data.unique),
               wrange = [-1,1],
               act = actf.sigmoid,
               layer_name = 'output',
               seed = 21,
               layer_type = 'output')
