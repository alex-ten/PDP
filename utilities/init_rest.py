import tensorflow as tf

def init_rest():
    all_vars = tf.global_variables()
    vars_to_init = [x for x in all_vars if not tf.is_variable_initialized(x).eval()]
    return tf.variables_initializer(vars_to_init)
