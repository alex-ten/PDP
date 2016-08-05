import tensorflow as tf
import os

def restore_xor(path, model):
    restore_dict = {'w_1': model['network'][0].W,
                    'b_1': model['network'][0].b,
                    'w_2': model['network'][1].W,
                    'b_2': model['network'][1].b}
    restore_op = tf.train.Saver(restore_dict)
    restore_op.restore(tf.get_default_session(), path)

def restore_params(path, layers, session):
    restore_dict = {}
    for layer in layers[1:-1]:
        restore_dict.setdefault(str(layer.W.name[:-2]), layer.W)
        restore_dict.setdefault(str(layer.b.name[:-2]), layer.b)
    print(restore_dict)
    restore_op = tf.train.Saver(restore_dict)
    session.run(restore_op.restore(session, path))

