import collections
import time
import pickle
import numpy as np
import tensorflow as tf
import FFBP.utilities.logger as logger
import FFBP.utilities.store_hyper_params as shp
from FFBP.artist.slider_plot import sum_figure
from FFBP.utilities.init_rest import init_rest
from FFBP.utilities.restore_params import restore_xor



class Network(object):
    def __init__(self, model):
        self.model = model
        self.dataset = None
        self.sess = tf.InteractiveSession()
        self.graph = tf.get_default_graph()
        self.logpath = logger.logdir()
        self.counter = 0
        self._loss = None
        self._opt = None
        self._below_ecrit = True
        self._lossHistory = np.empty(shape=(0, 2))
        self._settings = {}
        self._training = False

    def init_and_configure(self, learning_rate, momentum, loss):
        self.init_weights()
        self.configure(learning_rate, momentum, loss)

    def init_weights(self):
        # Initialize weights and biases
        Wb_vars = self.graph.get_collection('Wb')
        init_Wb_vars = tf.initialize_variables(Wb_vars)
        self.sess.run(init_Wb_vars)

    def configure(self, learning_rate, momentum, loss):
        self._loss = loss(self.model['labels'], self.model['network'][-1].activations)
        self._opt = tf.train.MomentumOptimizer(learning_rate, momentum)
        self._settings['opt_task'] = self._opt.minimize(self._loss)
        self._settings['saver'] = tf.train.Saver()
        self._settings['lrate'] = learning_rate
        self._settings['mrate'] = momentum
        self._settings['loss_func'] = loss
        for l in self.model['network']:
            # When run in current session tf.gradients returns numpy arrays with
            # batch_size number of rows and Layer.size number of columns.
            # That is, the rows of the returned arrays contain partial derivatives of loss with respect
            # to the argument tensor (here the loss tensor) of each unit in the layer given a particular input
            l.ded_netinp = tf.gradients(self._loss, l.netinp)
            l.ded_activations = tf.gradients(self._loss, l.activations)
            l.ded_W = tf.gradients(self._loss, l.W)
            l.ded_b = tf.gradients(self._loss, l.b)
        init = init_rest()
        self.sess.run(init)

    def restore(self, path, xor=True):
        # todo generalize this methods to enable restore of any set of variables
        if xor: restore_xor(path, model=self.model)

    def interact(self):
        pass

    def train(self, num_epochs, batch_size,
              ecrit = 0.01, tfcheckpoint = 100, permute = False):
        if not self._training:
            start = input("\n>>> Hit 'Enter' to begin training OR type in 'q' to terminate process ['Enter'/q]: ")
            print('    Now training...')
            self._training = True
            if start=='q':
                self._below_ecrit = False
                return
        if not self._below_ecrit:
            pass
        else:
            if num_epochs is None:
                usr_inp = input('\n>>> Enter number of epochs to train: ')
                num_epochs = int(usr_inp)

            hyper_parameters = [('Number of epochs:', num_epochs),
                                ('Learning rate:', self._settings['lrate']),
                                ('Momentum rate:', self._settings['mrate']),
                                ('Error:', self._settings['loss_func']),
                                ('Batch size:', batch_size),
                                ('Permuted mode:', permute)]
            shp.store_hyper_params(collections.OrderedDict(hyper_parameters), self.logpath)

            t0 = self.counter
            t1 = t0 + num_epochs
            global_start = time.time()

            for step in range(t0,t1):
                step_start = time.time()
                if permute: self.dataset.permute()

                train_dict = self.feed_dict(batch_size)
                _, loss_val = self.sess.run([self._settings['opt_task'], self._loss],
                                            feed_dict=train_dict)

                step_duration = time.time() - step_start

                # Collect stats (note that loss is measured before the gradients are applied):
                self._lossHistory = np.append(self._lossHistory, [[self.counter, loss_val]], axis=0)

                # Save a checkpoint periodically.
                if (self.counter + 1) % tfcheckpoint == 0 or (self.counter + 1) == t1:
                    self._settings['saver'].save(self.sess, self.logpath + '/tf_params/graph_vars_epoch-{}.ckpt'.format(self.counter))

                # Print something to stdout
                #print('Running epoch {}, loss: {}'.format(self.counter, loss_val)) #todo probably delete later
                if (step + 1) == t1:
                    training_duration = time.time() - global_start
                    print('Done training for {1}/{2} epochs ({0} seconds)\n'.format(round(training_duration, 3),
                                                                                  num_epochs,
                                                                                  self.counter+1))
                if loss_val < ecrit:
                    print('Reached critical loss value on epoch {}'.format(self.counter))
                    self._below_ecrit = False
                    break

                self.counter += 1

    def test(self, batch_size, evalfunc, snapshot=True, scope='all'):
        # Evaluate error defined by the user
        # Return values are parameters for self.snapshot() methods
        test_dict = self.feed_dict(batch_size)
        test = evalfunc(self.model['labels'], self.model['network'][-1].activations)

        # Evaluate test measure
        test_result = test.eval(feed_dict = test_dict)

        if scope=='all':
            scope = ['netinp', 'activations', 'W', 'b', 'ded_netinp', 'ded_activations', 'ded_W', 'ded_b']

        # Take a self-snapshot against a given input batch
        if snapshot:
            self.snapshot(scope, test_dict, test_result) # todo include error measure to the snapshot

        # Stdout
        if self._training: # . . . During training
            print('Test after epoch {}:'.format(self.counter))
            print('    Error tensor [{}] = {}'.format(test.name, test_result))
            go_on = input('\n>>> Continue training? [y/n]: ')
            while go_on != 'y' or go_on != 'n':
                if go_on == 'y':
                    break
                elif go_on == 'n':
                    self._below_ecrit = False
                    break
                else:
                    go_on = input("\n>>> Enter 'y' if you wish to proceed, or enter 'n' to terminate process [y/n]:" )
        else: #. . . . . . . . . . Before training
            print('Initial test:')
            print('    Error tensor [{}] = {}'.format(test.name, test_result))
            return scope, test_dict, test_result


    def snapshot(self, variables, batch, test_measure):
        # Create container for overall snapshot
        new_snap = {}
        for l in self.model['network']:
            # Create a container with pairs of keys and values for the inner dict
            metrix = zip(variables, self.sess.run(self.fetch(l, variables), feed_dict=batch))
            inner_dict = {}
            # Fill out the inner dict with keys and values
            for key, value in metrix:
                inner_dict[key] = logger.unroll(value, self.counter)
            # Label the inner dict with layer_name inside the snapshot
            new_snap[l.layer_name] = inner_dict
            new_snap['error'] = np.array([[self.counter, test_measure]])
        if not self._training:
            pickle.dump(new_snap, open(self.logpath + '/mpl_data/snapshot_log.pkl', 'wb'))
        else:
            with open(self.logpath + '/mpl_data/snapshot_log.pkl', 'rb') as opened_file:
                old_snap = pickle.load(opened_file)
            appended = logger.append_snapshot(old_snap, new_snap)
            pickle.dump(appended, open(self.logpath + '/mpl_data/snapshot_log.pkl', 'wb'))

    def visualize_loss(self):
        if self.counter > 0:
            def getybyx(y_vec, x):
                return y_vec[x]
            sum_figure(self._lossHistory, getybyx, 'epoch', 'loss', 'loss')

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

    def fetch(self, layer, scope):
        # Takes a layer object and returns a list of requested attributes
        basket = []
        for attribute in scope:
            basket.append(getattr(layer, attribute))
        return basket

    def off(self):
        self.sess.close()