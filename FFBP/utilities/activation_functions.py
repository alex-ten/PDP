import tensorflow as tf

def linear(x, name):
    return tf.identity(x, name=name)

def sigmoid(x, name):
    return tf.nn.sigmoid(x, name=name)

def tanh(x, name):
    return tf.nn.tanh(x, name=name)

def softmax(x, name):
    return tf.nn.softmax(x, name=name)

def relu(x, name):
    return tf.nn.relu(x, name=name)