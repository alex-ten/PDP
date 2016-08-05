# CPR
import tensorflow as tf
import time
import math
import collections
import FFBP.utilities.logdir as logdir
import FFBP.utilities.store_hyper_params as shp
from FFBP.utilities.restore_params import restore_xor

def SGD(model,
        dataset,
        num_epochs,
        learning_rate,
        momentum,
        error,
        batch_size,
        evaluation,
        checkpoint = 100,
        permute = False,
        _restore_XOR = False):

    # Create log directory to store hyperparameters, model parameters and events, return directory's path to dir_path
    dir_path = logdir.logdir()
    # Store hyperparameters in dir_path
    hyper_parameters = [('Number of epochs:', num_epochs),
                        ('Learning rate:', learning_rate),
                        ('Momentum rate:', momentum),
                        ('Error:', error),
                        ('Batch size:', batch_size),
                        ('Permuted mode:', permute)]
    shp.store_hyper_params(collections.OrderedDict(hyper_parameters), dir_path)

    # Tell TensorFlow that the model will be built into the default Graph.
    #with tf.Graph().as_default(): TODO what difference does this make?

    # Add ops for computing loss function and evatuation metric
    loss = error(model['labels'], model['network'][-1].activations)
    eval = evaluation(model['labels'], model['network'][-1].activations)

    # Define optimization algorithm.
    # Add ops for computing and applying gradients.
    # TODO: Add ops to obtain dEda and dEdnet
    opt = tf.train.MomentumOptimizer(learning_rate, momentum)
    dEda = 'gradients w.r.t. activations a.k.a. errors | tf.gradient(error_measure, )'
    dEdnet = 'gradients w.r.t. netinp a.k.a. deltas'
    dEdw = opt.compute_gradients(loss)
    learn = opt.apply_gradients(dEdw)

    # Build the summary operation based on the TF collection of Summaries.
    error_measure_summary = tf.scalar_summary(loss.name, loss)
    tss_summary = tf.scalar_summary(eval.name, eval)
    summary_op = tf.merge_all_summaries()

    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Use current session defined outside the scope of run_training() to run operations
    # Variable initialization must be executed outside run_training()
    current_session = tf.get_default_session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter(dir_path+'/events', current_session.graph)

    # And then after everything is built:
    current_session.run(init)

    if _restore_XOR is not False:
        restore_xor(_restore_XOR, model=model)

    # Start the training loop.
    global_start = time.time()
    for step in range(num_epochs):
        step_start = time.time()
        if permute: dataset.permute()

        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
        batch_xs, batch_ys = dataset.next_batch(batch_size, permute=permute)
        feed_dict = {model['images']: batch_xs, model['labels']: batch_ys}

        # Run one step of the training.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to current_session.run() and the value tensors will be
        # returned in the tuple from the call.
        _, loss_value, eval_value = current_session.run([learn, loss, eval],
                                                        feed_dict=feed_dict)

        step_duration = time.time() - step_start

        # Write the summaries and print an overview fairly often.
        summary_str = current_session.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        if num_epochs <= 10:
            # Print status to stdout.
            print('Step {0}: tss = {1} ({2} sec)'.format(step, eval_value, round(step_duration,5)))
            # Update the events file.
        else:
            if step % math.floor(num_epochs/10) == 0:
                print('Step {0}: tss = {1} ({2} sec)'.format(step, eval_value, round(step_duration, 5)))
        # Save a checkpoint and evaluate the model periodically.
        if (step + 1) % checkpoint == 0 or (step + 1) == num_epochs:
            saver.save(current_session, dir_path+'/params/graph_vars_epoch-{}.ckpt'.format(step), global_step=step)
            # Evaluate against the training set.
            '''
            print('Training Data Eval:')
            do_eval(current_session,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.train)
            # Evaluate against the validation set.
            print('Validation Data Eval:')
            do_eval(current_session,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.validation)
            # Evaluate against the test set.
            print('Test Data Eval:')
            do_eval(current_session,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.test)
            '''
            if (step + 1) == num_epochs:
                training_duration = time.time() - global_start
                print('Training complete in ≈ {} seconds'.format(round(training_duration,3)))
                print('Final {}: {}'.format(loss.name , eval_value))
                print('\ntensorboard --logdir={}/events'.format(dir_path))