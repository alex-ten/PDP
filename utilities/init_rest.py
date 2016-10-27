import tensorflow as tf

def init_rest():
    all_vars = tf.all_variables()
    vars_to_init = [x for x in all_vars if not tf.is_variable_initialized(x).eval()]
    return tf.initialize_variables(vars_to_init)
