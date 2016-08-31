import collections
import time
import pickle
import numpy as np
import tensorflow as tf
import FFBP.utilities.logger as logger
import FFBP.utilities.store_hyper_params as shp
from FFBP.classes.LayerLog import LayerLog
from FFBP.utilities.init_rest import init_rest
from FFBP.utilities.restore_params import restore_xor
from FFBP.visualization.visual_error import sum_figure
import FFBP.visualization.Artist as vc



class Network(object):
    def __init__(self, model, name='NN'):
        self.name = name
        self.model = model
        self.dataset = None # todo feed dataset externally
        self.sess = tf.InteractiveSession()
        self.graph = tf.get_default_graph()
        self.logpath = logger.logdir()
        self.counter = 0
        self._loss = None
        self._opt = None
        self._lossHistory = np.empty(shape=(0, 2))
        self._settings = {}
        self._interactive = False
        self._training = False
        self._terminate = False

    def init_and_configure(self, learning_rate, momentum, loss):
        self.init_weights()
        self.configure(learning_rate, momentum, loss)

    def init_weights(self):
        # Initialize weights and biases
        Wb_vars = self.graph.get_collection('Wb')
        init_Wb_vars = tf.initialize_variables(Wb_vars)
        self.sess.run(init_Wb_vars)

    def configure(self,
                  loss,
                  train_batch_size,
                  test_batch_size,
                  learning_rate=0.5,
                  momentum=0.9,
                  permute=False,
                  ecrit=0.01,
                  test_func=None,
                  test_scope='all'):
        if test_scope != 'all':
            raise UserWarning("snapshot_plotter will not be able to show test data correctly. Set test_scope='all' to visualize snapshots")
        self._loss = loss(self.model['labels'], self.model['network'][-1].activations)
        self._opt = tf.train.MomentumOptimizer(learning_rate, momentum)
        self._settings['loss_func'] = loss
        self._settings['train_batch'] = train_batch_size
        self._settings['test_batch'] = test_batch_size
        self._settings['lrate'] = learning_rate
        self._settings['mrate'] = momentum
        self._settings['permute'] = permute
        self._settings['ecrit'] = ecrit
        self._settings['test_func'] = test_func
        self._settings['scope'] = test_scope
        self._settings['opt_task'] = self._opt.minimize(self._loss)
        self._settings['saver'] = tf.train.Saver()
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

    def interact(self, train_set, test_set, take_snapshots=True):
        self._interactive = True
        fill = len(self.name)
        print('\n[{}] Now in interactive mode...'.format(self.name))
        print('[{}] Input your next action:'.format(self.name))
        print(' '*fill +' '*5+ "• # of epochs to train")
        print(' '*fill +' '*5+ "• 't' to test")
        print(' '*fill +' '*5+ "• 'c' to train until ecrit is reached")
        print(' '*fill +' '*5+ "• 'q' to quit")
        action = input('#/t/c/q -> ')
        while not self._terminate:
            if action == 'q':
                self._terminate = True
                self.off()
                print('[{}] Process terminated.'.format(self.name))
                break
            try:
                usr_inp = int(action)
                self.run_training(num_epochs = usr_inp,
                                  dataset = train_set,
                                  batch_size = self._settings['train_batch'])
                if self._terminate:
                    print('[{}] Would you like to test before terminating the process?'.format(self.name))
                    action = input('y/n -> ')
                    while not any([action=='y', action=='n']):
                        print("[{}] Choose one of the folloing options:".format(self.name))
                        action = input('y/n ->')
                    if action=='y':
                        self.test(dataset = test_set,
                                  batch_size = self._settings['test_batch'],
                                  evalfunc = self._settings['test_func'],
                                  snapshot = take_snapshots)
                        self.off()
                        print('[{}] Process terminated.'.format(self.name))
                        break
                    elif action=='n':
                        self.off()
                        print('[{}] Process terminated.'.format(self.name))
                        break
                print('[{}] Input your next action:'.format(self.name))
                action = input('#/t/c/q -> ')
            except ValueError:
                if action=='t':
                    self.test(dataset = test_set,
                              batch_size = self._settings['test_batch'],
                              evalfunc = self._settings['test_func'],
                              snapshot = take_snapshots)
                    print('[{}] Input your next action:'.format(self.name))
                    action = input('#/t/c/q -> ')
                elif action=='c':
                    self.run_training(num_epochs =1000 ** 2,
                                      dataset = train_set,
                                      batch_size = self._settings['train_batch'])

                else:
                    print("[{}] Choose one of the following options:".format(self.name))
                    action = input('#/t/c/q -> ')

                if self._terminate:
                    print('[{}] Would you like to test before terminating the process?'.format(self.name))
                    action = input('y/n -> ')
                    while not any([action == 'y', action == 'n']):
                        print("[{}] Choose one of the folloing options:".format(self.name))
                        action = input('y/n ->')
                    if action == 'y':
                        self.test(dataset = test_set,
                                  batch_size = self._settings['test_batch'],
                                  evalfun = self._settings['test_func'],
                                  snapshot=take_snapshots)
                        self.off()
                        print('[{}] Process terminated.'.format(self.name))
                        break
                    elif action == 'n':
                        self.off()
                        print('[{}] Process terminated.'.format(self.name))
                        break
        self._interactive = False

    def tnt(self, max_epochs, train_set, test_set, snp_checkpoint=100, tf_checkpoint = 100):
        print('[{}] Now in train and test mode...'.format(self.name))
        while self.counter < max_epochs:
            score, _, __ = self.test(test_set, self._settings['test_batch'], evalfunc=self._settings['test_func'])
            print('[{}] epoch {}: {}'.format(self.name, self.counter, score))
            self.run_training(snp_checkpoint, train_set, self._settings['train_batch'], ecrit=self._settings['ecrit'], tf_checkpoint=tf_checkpoint)
            if self._terminate:
                score, _, __ = self.test(test_set, self._settings['test_batch'], evalfunc=self._settings['test_func'])
                print('[{}] Final error: {}'.format(self.name, score))
                print('[{}] Process terminated.'.format(self.name))
                self.off()
                break

    def run_training(self, num_epochs, dataset, batch_size, ecrit = 0.01, tf_checkpoint = 100):
        if not self._training:
            # perform on the first epoch
            self._training = True
            hyper_parameters = [('Number of epochs:', num_epochs),
                                ('Learning rate:', self._settings['lrate']),
                                ('Momentum rate:', self._settings['mrate']),
                                ('Error:', self._settings['loss_func']),
                                ('Batch size:', batch_size),
                                ('Permuted mode:', self._settings['permute'])]
            shp.store_hyper_params(collections.OrderedDict(hyper_parameters), self.logpath)
        if self._terminate:
            return
        else:
            self.show('[{}] Now training...'.format(self.name))
            t0 = self.counter
            t1 = t0 + num_epochs
            global_start = time.time()

            for step in range(t0,t1):
                step_start = time.time()
                if self._settings['permute']==True: dataset.permute()

                train_dict = self.feed_dict(dataset, batch_size)
                _, loss_val = self.sess.run([self._settings['opt_task'], self._loss],
                                            feed_dict=train_dict)

                step_duration = time.time() - step_start

                # Collect stats (note that loss is measured before the gradients are applied):
                self._lossHistory = np.append(self._lossHistory, [[self.counter, loss_val]], axis=0)

                # Save a checkpoint periodically.
                if (self.counter + 1) % tf_checkpoint == 0 or (self.counter + 1) == t1:
                    self._settings['saver'].save(self.sess, self.logpath + '/tf_params/graph_vars_epoch-{}.ckpt'.format(self.counter))

                if loss_val < ecrit:
                    print('[{}] Reached critical loss value on epoch {}'.format(self.name, self.counter))
                    self._terminate = True
                    break

                # Print something to stdout
                if (step + 1) == t1:
                    training_duration = time.time() - global_start
                    self.show('[{}] Done training for {}/{} epochs ({} seconds)'.format(self.name,
                                                                                    num_epochs,
                                                                                    (self.counter + 1),
                                                                                    round(training_duration, 3)))
                self.counter += 1

    def test(self, dataset, batch_size, evalfunc, snapshot=True):
        # Evaluate error defined by the user
        # Return values are parameters for self.snapshot() methods
        test_dict = self.feed_dict(dataset, batch_size)
        test = evalfunc(self.model['labels'], self.model['network'][-1].activations)

        # Evaluate test measure
        test_result = test.eval(feed_dict = test_dict)
        if self._settings['scope']=='all':
            self._settings['scope'] = ('inp', 'netinp', 'activations', 'W', 'b', 'ded_netinp', 'ded_activations', 'ded_W', 'ded_b')

        # Take a self-snapshot against a given input batch
        if snapshot:
            self.snapshot(self._settings['scope'], test_dict, test_result)

        # Stdout
        if self._training: # . . . During training
            self.show('[{}] Test after epoch {}:'.format(self.name, self.counter))
            self.show('[{}] Error tensor |{}| = {}'.format(self.name, test.name, test_result))
        else: #. . . . . . . . . . Before training
            self.show('[{}] Initial test...'.format(self.name))
            self.show('[{}] Error tensor [{}] = {}'.format(self.name, test.name, test_result))
        # self.visualize_layers()
        return test_result, self._settings['scope'], test_dict

    def snapshot(self, attributes, batch, test_measure):
        try:
            with open(self.logpath + '/mpl_data/snapshot_log.pkl', 'rb') as file:
                snap = pickle.load(file)
            snap['checkpoints'] = np.append(snap['checkpoints'], [self.counter], axis=0)
            snap['error'] = np.append(snap['error'], [test_measure], axis=0)
            for l in self.model['network']:
                state = zip(attributes, self.sess.run(self.fetch(l, attributes), feed_dict=batch))
                snap[l.layer_name].append(state)
            pickle.dump(snap, open(self.logpath + '/mpl_data/snapshot_log.pkl', 'wb'))
        except FileNotFoundError:
            new_snap = collections.OrderedDict({'checkpoints': np.array([self.counter], dtype=int),
                        'error': np.array([test_measure], dtype=float), 'attributes': attributes})
            for l in self.model['network']:
                log = LayerLog(l)
                state = zip(attributes, self.sess.run(self.fetch(l, attributes), feed_dict=batch))
                log.append(state)
                new_snap[l.layer_name] = log
            pickle.dump(new_snap, open(self.logpath + '/mpl_data/snapshot_log.pkl', 'wb'))






    def visualize_loss(self):
        if self.counter > 0:
            def getybyx(y_vec, x):
                return y_vec[x]
            sum_figure(self._lossHistory, getybyx, 'epoch', 'loss', 'loss')

    def visualize_layers(self, pattern=0):
        snap = vc.NetworkData(self.logpath+'/mpl_data/snapshot_log.pkl')
        plot = vc.Artist(style_sheet='seaborn-dark')
        plot.outline_all(snap)
        plot.fill_axes(snap, self.counter, c='coolwarm', pattern=pattern)
        plot.remove_ticklabels()
        plot.show()

    def feed_dict(self, dataset, batch_size):
        # Fill a feed dictionary with the actual set of images and labels
        # for current training step.
        # Takes attribute self.dataset as a default data set

        batch_xs, batch_ys = dataset.next_batch(batch_size)
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

    def show(self, string):
        if self._interactive:
            print(string)

    def off(self):
        self.sess.close()