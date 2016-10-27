import tensorflow as tf

def mask(x, seq_lengths, max_len, batch_size):
    # Make a (batch_size x max_len) matrix where each row contains the length repeated max_len times.
    lengths_transposed = tf.expand_dims(seq_lengths, 1, name = 'seq_lengths')
    print(lengths_transposed)
    lengths_tiled = tf.tile(lengths_transposed, [1, max_len])
    # Unroll
    lengths_unrolled = tf.reshape(lengths_tiled, [1, -1])

    # Make a (batch_size x max_len) matrix where each row contains [0, 1, ..., max_len - 1]
    range = tf.range(0, max_len, 1)
    range_row = tf.expand_dims(range, 0)
    range_tiled = tf.tile(range_row, [batch_size, 1])
    range_unrolled = tf.reshape(range_tiled, [1, -1])

    # Use the logical operations to create a mask
    bool_mask = tf.less(range_unrolled, lengths_unrolled)

    # Use the select operation to select between 1 or 0 for each value.
    int_mask = tf.cast(bool_mask, tf.float32)

    return x * tf.transpose(int_mask)

def main():
    import numpy as np
    np.set_printoptions(precision=2, suppress=True)

    num_steps = 12
    batch_size = 3
    data_dim = 4
    seq_lens = [12, 8, 3]

    sess = tf.InteractiveSession()

    dummy = tf.random_uniform((num_steps * batch_size, data_dim))
    print(dummy.eval())

    masked_dummy = mask(dummy, seq_lens, num_steps, batch_size)
    print(masked_dummy.eval())

if __name__=='__main__': main()


