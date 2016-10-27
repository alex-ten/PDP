import tensorflow as tf

def clipped(x):
    # this handles cases when y * tf.log(y') outputs NaN
    return tf.clip_by_value(x, 1e-10, 1.0)

def cross_entropy(target, activation):
    return -tf.reduce_sum(target * tf.log(clipped(activation)) + (1 - target) * tf.log(clipped(1 - activation)),
                          name='cross_entropy')

def squared_error(target, activation):
    return tf.reduce_sum(tf.squared_difference(target, activation) / 2, name='squared_error')
