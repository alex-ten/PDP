# Consult wikipedia for overview of these functions:
# https://en.wikipedia.org/wiki/Activation_function
import tensorflow as tf


def linear(x, name='linear'):
    return tf.identity(x, name=name)

def sigmoid(x, name='sigmoid'):
    return tf.nn.sigmoid(x, name=name)

def tanh(x, name='tanh'):
    return tf.nn.tanh(x, name=name)

def softmax(x, name='softmax'):
    return tf.nn.softmax(x, name=name)

def softplus(x, name='softplus'):
    return tf.nn.softplus(x, name=name)

def relu(x, name='relu'):
    return tf.nn.relu(x, name=name)

def softsign(x, name='softsign'):
    return tf.nn.softsign(x, name=name)