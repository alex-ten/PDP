import tensorflow as tf
from tensorflow.python.ops.variables import report_uninitialized_variables
import time
import math
import collections
import numpy as np
import FFBP.utilities.logdir as logdir
import FFBP.utilities.store_hyper_params as shp
from FFBP.utilities.restore_params import restore_xor
from FFBP.utilities.init_rest import init_rest

class Network(object):
    def __init__(self, model):
        self.model = model
        self.dataset = None
        self.sess = tf.InteractiveSession()
        self.graph = tf.get_default_graph()
        self.logpath = logdir.logdir()
        self.counter = 0
        self._loss = None
        self._opt = None

    def init_weights(self):
        # Initialize weights and biases
        Wb_vars = self.graph.get_collection('Wb')
        init_Wb_vars = tf.initialize_variables(Wb_vars)
        self.sess.run(init_Wb_vars)


    def restore(self, path, xor=True):
        if xor: restore_xor(path, model=self.model)

    def f_prop(self):
        pass

    def b_prop(self):
        pass

    def train(self,
        num_epochs,
        learning_rate,
        momentum,
        error,
        batch_size,
        checkpoint = 100,
        permute = False,):

        if num_epochs is None:
            usr_inp = input('Enter number of epochs to train: ')
            num_epochs = int(usr_inp)

        hyper_parameters = [('Number of epochs:', num_epochs),
                            ('Learning rate:', learning_rate),
                            ('Momentum rate:', momentum),
                            ('Error:', error),
                            ('Batch size:', batch_size),
                            ('Permuted mode:', permute)]
        shp.store_hyper_params(collections.OrderedDict(hyper_parameters), self.logpath)

        self._loss = error(self.model['labels'], self.model['network'][-1].activations)
        opt = tf.train.MomentumOptimizer(learning_rate, momentum)
        learn = opt.minimize(self._loss)
        error_measure_summary = tf.scalar_summary(self._loss.name, self._loss)
        summary_op = tf.merge_all_summaries()

        saver = tf.train.Saver()

        summary_writer = tf.train.SummaryWriter(self.logpath + '/events', self.sess.graph)

        init = init_rest()
        self.sess.run(init)

        t0 = self.counter
        t1 = t0 + num_epochs
        global_start = time.time()

        for step in range(t0,t1):
            print('performing step {}'.format(step))
            step_start = time.time()
            if permute: self.dataset.permute()

            train_dict = self.feed_dict(batch_size)

            _, loss_value, summary_str = self.sess.run([learn, self._loss, summary_op], feed_dict=train_dict)

            step_duration = time.time() - step_start

            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

            # Save a checkpoint periodically.
            if (self.counter + 1) % checkpoint == 0 or (self.counter + 1) == t1:
                saver.save(self.sess, self.logpath + '/params/graph_vars_epoch-{}.ckpt'.format(self.counter))

            if (step + 1) == t1:
                training_duration = time.time() - global_start
                print('Done training for {1}/{2} epochs ({0} seconds)'.format(round(training_duration, 3),
                                                                              num_epochs,
                                                                              self.counter+1))
            self.counter += 1

    def test(self, batch_size):
        np.set_printoptions(precision=5,suppress=True)
        test_dict = self.feed_dict(batch_size)
        # Define all gradients for each layer
        ded_netinp = tf.gradients(self._loss, [x.netinp for x in self.model['network']])
        ded_activations = tf.gradients(self._loss, [x.activations for x in self.model['network']])
        # when run in current session dEdnet and dEda return numpy arrays with
        # batch_size number of rows and Layer.size number of columns.
        # Basically, the rows contain partial derivatives of loss with respect
        # to either netinp or activation of each unit in the layer given a particular input
        ded_W = tf.gradients(self._loss, [x.W.ref() for x in self.model['network']])
        ded_b = tf.gradients(self._loss, [x.b.ref() for x in self.model['network']])
        # dEdw and dEdb return a lists of (gradient, variable) pairs.
        # E.g. [(dEdw_1, hidden1.W),(dEdw_2, output.W)]
        netinp, activations, W, b = self.sess.run([ded_netinp, ded_activations, ded_W, ded_b], feed_dict=test_dict)
        print('Partial derivatives w.r.t. net inputs:\n', netinp)
        print('Partial derivatives w.r.t. activations:\n', activations)
        print('Partial derivatives w.r.t. weights:\n', W)
        print('Partial derivatives w.r.t. biases:\n', b)

    def feed_dict(self, batch_size):
        # Fill a feed dictionary with the actual set of images and labels
        # for current training step.
        # Takes attribute self.dataset as a default data set
        batch_xs, batch_ys = self.dataset.next_batch(batch_size)
        feed_dict = {}
        # if len(model['images']) > 1:
        start = 0
        end = 0
        for in_placeholder in self.model['images']:
            end += int(in_placeholder.get_shape()[1])
            feed_dict[in_placeholder] = batch_xs[:, start:end]
            start += end
        feed_dict[self.model['labels']] = batch_ys
        return feed_dict

    def print_logdir(self):
        print('\ntensorboard --logdir={}/events'.format(self.logpath))

    def off(self):
        self.sess.close()