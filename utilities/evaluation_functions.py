import tensorflow as tf


def clipped(x):
    # this handles cases when y * tf.log(y') outputs NaN
    return tf.clip_by_value(x, 1e-10, 1.0)

def tce(target, activation):
    return tf.reduce_sum(target * tf.log(clipped(activation)) + (1 - target) * tf.log(clipped(1 - activation)),
                          name='tce')

def tss(target, activation):
    return tf.reduce_sum(tf.squared_difference(target, activation), name='tss')