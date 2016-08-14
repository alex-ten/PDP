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
        self.loss = None
        self.ecrit_not_reached = True
        self._training = False

    def init_weights(self):
        # Initialize weights and biases
        Wb_vars = self.graph.get_collection('Wb')
        init_Wb_vars = tf.initialize_variables(Wb_vars)
        self.sess.run(init_Wb_vars)


    def restore(self, path, xor=True):
        if xor: restore_xor(path, model=self.model)

    def train(self,
        num_epochs,
        learning_rate,
        momentum,
        loss,
        batch_size,
        ecrit = 0.01,
        checkpoint = 100,
        permute = False):

        self._training = True

        if num_epochs is None:
            usr_inp = input('Enter number of epochs to train: ')
            num_epochs = int(usr_inp)

        hyper_parameters = [('Number of epochs:', num_epochs),
                            ('Learning rate:', learning_rate),
                            ('Momentum rate:', momentum),
                            ('Error:', loss),
                            ('Batch size:', batch_size),
                            ('Permuted mode:', permute)]
        shp.store_hyper_params(collections.OrderedDict(hyper_parameters), self.logpath)

        self.loss = loss(self.model['labels'], self.model['network'][-1].activations)
        opt = tf.train.MomentumOptimizer(learning_rate, momentum)
        learn = opt.minimize(self.loss)
        error_measure_summary = tf.scalar_summary(self.loss.name, self.loss)
        summary_op = tf.merge_all_summaries()

        saver = tf.train.Saver()

        summary_writer = tf.train.SummaryWriter(self.logpath + '/events', self.sess.graph)

        init = init_rest()
        self.sess.run(init)

        t0 = self.counter
        t1 = t0 + num_epochs
        global_start = time.time()

        for step in range(t0,t1):
            print('Running epoch {}'.format(self.counter))
            step_start = time.time()
            if permute: self.dataset.permute()

            train_dict = self.feed_dict(batch_size)

            _, loss_val, summary_str = self.sess.run([learn, self.loss, summary_op], feed_dict=train_dict)

            step_duration = time.time() - step_start

            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

            # Save a checkpoint periodically.
            if (self.counter + 1) % checkpoint == 0 or (self.counter + 1) == t1:
                saver.save(self.sess, self.logpath + '/params/graph_vars_epoch-{}.ckpt'.format(self.counter))

            if (step + 1) == t1:
                training_duration = time.time() - global_start
                print('Done training for {1}/{2} epochs ({0} seconds)\n'.format(round(training_duration, 3),
                                                                              num_epochs,
                                                                              self.counter+1))
            if loss_val < ecrit:
                print('Reached critical loss value on epoch {}'.format(self.counter))
                self.ecrit_not_reached = False
                break

            self.counter += 1

    def test(self, batch_size, eval, loss):
        np.set_printoptions(precision=5,suppress=True)
        test_dict = self.feed_dict(batch_size)

        if self.loss is None:
            self.loss = loss(self.model['labels'], self.model['network'][-1].activations)
        test = eval(self.model['labels'], self.model['network'][-1].activations)

        # When run in current session tf.gradients returns numpy arrays with
        # batch_size number of rows and Layer.size number of columns.
        # Basically, the rows contain partial derivatives of loss with respect
        # to the argument tensor of each unit in the layer given a particular input

        ded_netinp = tf.gradients(self.loss, [x.netinp for x in self.model['network']])
        ded_activations = tf.gradients(self.loss, [x.activations for x in self.model['network']])
        ded_W = tf.gradients(self.loss, [x.W.ref() for x in self.model['network']])
        ded_b = tf.gradients(self.loss, [x.b.ref() for x in self.model['network']])

        test_result, netinp, activations, W, b = self.sess.run([test, ded_netinp, ded_activations, ded_W, ded_b],
                                                               feed_dict=test_dict)

        print('Testing network after epoch {}:'.format(self.counter))
        print('Error tensor [{}]:  {}'.format(test.name, test_result))
        print('Partial derivatives w.r.t. net inputs:\n', netinp)
        print('Partial derivatives w.r.t. activations:\n', activations)
        print('Partial derivatives w.r.t. weights:\n', W)
        print('Partial derivatives w.r.t. biases:\n', b)

        if self._training:
            go_on = input('continue training? [y/n]: ')
            if go_on == 'y':
                pass
            elif go_on == 'n':
                self.ecrit_not_reached = False

    def feed_dict(self, batch_size):
        # Fill a feed dictionary with the actual set of images and labels
        # for current training step.
        # Takes attribute self.dataset as a default data set

        batch_xs, batch_ys = self.dataset.next_batch(batch_size)
        feed_dict = {}
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