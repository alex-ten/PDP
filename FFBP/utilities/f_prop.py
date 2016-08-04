import tensorflow as tf

def f_prop(image, ):
    # Define init operation
    init_op = tf.initialize_all_variables()

    # Run the graph
    with tf.Session() as sess:
        sess.run(init_op)
        R.restore(sess, "xor_files/xor-params.ckpt")
        tf.get_default_graph()
        batch_xs, batch_ys = DS.next_batch(4)
        print(sess.run(output.activations, feed_dict={x: batch_xs, y_: batch_ys}))
        summary_str = sess.run(merged, feed_dict={x: batch_xs, y_: batch_ys})
        summary_writer.add_summary(summary_str, 0)
        summary_writer.flush()
        print('\ntensorflow --logdir={}/events'.format(dir_path))