import collections
import time
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tkinter as tk

import FFBP.utilities.logger as logger
from FFBP.utilities.store_configurations import store
from FFBP.utilities.init_rest import init_rest
from FFBP.utilities.restore_params import restore_xor
from FFBP.utilities.printProgress import printProgress
from FFBP.visualization.NetworkData import NetworkData
from FFBP.visualization.VisErrorApp import VisErrorApp
from FFBP.visualization.VisLayersApp import VisLayersApp


class Network(object):
    def __init__(self, model, name='NN'):
        self.name = name
        self.model = model
        self.sess = tf.InteractiveSession()
        self.graph = tf.get_default_graph()
        self.logpath = logger.logdir()
        self.counter = 0
        self.train_set = None
        self.test_set = None
        self._loss = None
        self._opt = None
        self._settings = {}
        self._lossHistory = []
        self._last_test = None
        self._interactive = False
        self._training = False
        self._terminate = False
        self._errVisApp = None
        self._layVisApp = None
        self._vis_app_settings = {'ppc': 30,
                                  'dpi': 96}

    def init_and_configure(self,
                  loss,
                  train_batch_size,
                  learning_rate=0.5,
                  momentum=0.9,
                  permute=False,
                  ecrit=0.01,
                  test_func=None,
                  test_scope='all'):
        self.init_weights()
        self.configure(loss,train_batch_size, learning_rate,
                       momentum, permute, ecrit, test_func)

    def init_weights(self):
        # Initialize weights and biases
        Wb_vars = self.graph.get_collection('Wb')
        init_Wb_vars = tf.initialize_variables(Wb_vars)
        self.sess.run(init_Wb_vars)

    def configure(self,
                  loss,
                  train_batch_size,
                  learning_rate=0.5,
                  momentum=0.9,
                  permute=False,
                  ecrit=0.01,
                  test_func=None,
                  test_scope='all'):
        if test_scope != 'all':
            input('''Warning: current visualizer will not be able to show test data correctly.
Set test_scope='all' to visualize snapshots. If you want to continue, press enter.''')
        self._loss = loss(self.model['labels'], self.model['network'][-1].act)
        self._opt = tf.train.MomentumOptimizer(learning_rate, momentum)
        self._settings['loss_func'] = loss
        self._settings['train_batch'] = train_batch_size
        self._settings['lrate'] = learning_rate
        self._settings['mrate'] = momentum
        self._settings['permute'] = permute
        self._settings['ecrit'] = ecrit
        self._settings['test_func'] = test_func
        self._settings['scope'] = test_scope
        self._settings['opt_task'] = self._opt.minimize(self._loss)
        self._settings['saver'] = tf.train.Saver()
        for l in self.model['network']:
            # When run in current session tf.gradients returns a list of numpy arrays with
            # batch_size number of rows and Layer.size number of columns.
            # That is, the rows of the returned arrays contain partial derivatives of loss with respect
            # to the argument tensor (here the loss tensor) of each unit in the layer given a particular input
            l.ded_net = tf.gradients(self._loss, l.net)
            l.ded_act = tf.gradients(self._loss, l.act)
            l.ded_W = tf.gradients(self._loss, l.W)
            l.ded_b = tf.gradients(self._loss, l.b)
        hyper_parameters = [('Learning rate:', self._settings['lrate']),
                            ('Momentum rate:', self._settings['mrate']),
                            ('Error:', self._settings['loss_func']),
                            ('Batch size:', self._settings['train_batch']),
                            ('Permuted mode:', self._settings['permute']),
                            ('Ecrit:', self._settings['ecrit'])]
        store(collections.OrderedDict(hyper_parameters), self.logpath)
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
        print(' '*fill +' '*5+ "* # of epochs to train")
        print(' '*fill +' '*5+ "* 't' to test")
        print(' '*fill +' '*5+ "* 'c' to train until ecrit is reached")
        print(' '*fill +' '*5+ "* 'q' to quit")
        action = input('#/t/c/q -> ')
        while not self._terminate:
            if action == 'q':
                self._terminate = True
                print('[{}] Process terminated.'.format(self.name))
                break
            try:
                usr_inp = int(action)
                self._train(num_epochs = usr_inp,
                            dataset = train_set,
                            batch_size = self._settings['train_batch'])
                if self._terminate:
                    print('[{}] Would you like to test before terminating the process?'.format(self.name))
                    action = input('y/n -> ')
                    while not any([action=='y', action=='n']):
                        print("[{}] Choose one of the folloing options:".format(self.name))
                        action = input('y/n ->')
                    if action=='y':
                        self._test(dataset = test_set,
                                   evalfunc = self._settings['test_func'],
                                   snapshot = take_snapshots)
                        self.visualize_error()
                        print('[{}] Process terminated.'.format(self.name))
                        break
                    elif action=='n':
                        print('[{}] Process terminated.'.format(self.name))
                        break
                print('[{}] Input your next action:'.format(self.name))
                action = input('#/t/c/q -> ')
            except ValueError:
                if action=='t':
                    self._test(dataset = test_set,
                               evalfunc = self._settings['test_func'],
                               snapshot = take_snapshots)
                    self.visualize_error()
                    print('[{}] Input your next action:'.format(self.name))
                    action = input('#/t/c/q -> ')
                elif action=='c':
                    self._train(num_epochs =1000 ** 2,
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
                        action = input('y/n -> ')
                    if action == 'y':
                        self._test(dataset = test_set,
                                   evalfunc = self._settings['test_func'],
                                   snapshot=take_snapshots)
                        self.visualize_error()
                        print('[{}] Process terminated.'.format(self.name))
                        break
                    elif action == 'n':
                        print('[{}] Process terminated.'.format(self.name))
                        break
        self._interactive = False

    def tnt(self, max_epochs, test_freq=0, ckpt_freq = 0, **kwargs):
        print(self._interactive)
        if 'train_set' in kwargs: train_set = kwargs['train_set']
        else: train_set = self.train_set
        if 'test_set' in kwargs: test_set = kwargs['test_set']
        else: test_set = self.test_set
        print('[{}] Now in train and test mode...'.format(self.name))
        while self.counter < max_epochs:
            result = self._test(test_set, evalfunc=self._settings['test_func'], snapshot=self._checkLastTest())
            print('[{}] epoch {}: {}'.format(self.name, self.counter, result))
            self._train(test_freq, train_set, self._settings['train_batch'], ecrit=self._settings['ecrit'], ckpt_freq=ckpt_freq)
            if self._terminate or self.counter == max_epochs:
                result = self._test(test_set, evalfunc=self._settings['test_func'], snapshot=self._checkLastTest())
                print('[{}] Final error (epoch {}): {}'.format(self.name, self.counter, result))
                print('[{}] Process terminated.'.format(self.name))
                break

    def train(self, num_epochs = 1, vis = False, ckpt_freq = False):
        self._interactive = True
        if ckpt_freq:
            freq = ckpt_freq
            assert type(freq) is int, ValueError('Expected an integer, got {}'.format(type(freq)))
        self._train(num_epochs,
                    self.train_set,
                    self._settings['train_batch'],
                    self._settings['ecrit'],
                    ckpt_freq)
        if vis: self.visualize_error(
            str(self._settings['test_func']).split(' ')[1])
        self._interactive = False

    def test(self, vis = False):
        self._interactive = True
        self._test(dataset = self.test_set,
                   evalfunc = self._settings['test_func'],
                   snapshot = self._checkLastTest())
        if vis:
            try:
                self.visualize_layers()
            except tk._tkinter.TclError:
                self._layVisApp = None
                self.visualize_layers()
        self._interactive = False

    def visualize_error(self, error_name = 'error'):
        if self._errVisApp is None:
            root1 = tk.Tk()
            figure = plt.figure(1, facecolor='w', dpi=self._vis_app_settings['dpi'])
            self._errVisApp = VisErrorApp(root1,
                                          figure,
                                          self._lossHistory,
                                          'epoch',
                                          error_name,
                                          error_name)
        if self.counter > 0:
            self._errVisApp.catch_up(self._lossHistory)

    def visualize_layers(self):
        snap = NetworkData(self.logpath + '/mpl_data/snapshot_log.pkl')
        if self._layVisApp is None:
            root2 = tk.Tk()
            self._layVisApp = VisLayersApp(root2, snap)
        else:
            self._layVisApp.catch_up(snap)

    def reset(self):
        self.sess = tf.InteractiveSession()
        self.graph = tf.get_default_graph()
        self.counter = 0
        self._lossHistory = []
        self._interactive = False
        self._training = False
        self._terminate = False
        self._errVisApp = None
        self._layVisApp = None

    def _train(self, num_epochs, dataset, batch_size, ecrit=0.01, ckpt_freq=100, permute = False):
        if not self._training: self._training = True
        if self._terminate:
            return
        else:
            if 'permute' in self._settings:
                permute = self._settings['permute']
            t0 = self.counter
            t1 = t0 + num_epochs
            start = time.time()
            self._inBar(t0, t1)
            for step in range(t0, t1):
                self._inBar(step, t1 - 1)
                if permute: dataset.permute()
                train_dict, _ = self._feed_dict(dataset, batch_size=batch_size)
                _, loss_val = self.sess.run([self._settings['opt_task'], self._loss],
                                            feed_dict=train_dict)

                # Collect stats (note that loss is measured before the gradients are applied):
                self._lossHistory.append(loss_val)

                # Save a checkpoint periodically.
                if ckpt_freq:
                    if (self.counter + 1) % ckpt_freq == 0 or (self.counter + 1) == t1:
                        self._settings['saver'].save(self.sess,
                                                     self.logpath + '/tf_params/graph_vars_epoch-{}.ckpt'.format(
                                                         self.counter))

                if loss_val < ecrit:
                    self._inBar(1, 1)
                    print('[{}] Reached critical loss value on epoch {}'.format(self.name, self.counter))
                    self._terminate = True
                    break

                # Print something to stdout
                if (step + 1) == t1:
                    training_duration = time.time() - start
                    self._inPrint('[{}] Done training for {}/{} epochs ({} seconds)'.format(self.name,
                                                                                           num_epochs,
                                                                                           (self.counter + 1),
                                                                                           round(training_duration,3)))
                self.counter += 1

    def _test(self, dataset, evalfunc, snapshot=False):
        test_dict, inp_vects = self._feed_dict(dataset)
        inp_names = dataset.names
        test = evalfunc(self.model['labels'], self.model['network'][-1].act)

        # Evaluate test measure
        test_result = test.eval(feed_dict=test_dict)
        if 'scope' in self._settings:
            scope = self._settings['scope']
        else: scope = snapshot
        if scope == 'all':
            scope = ['inp', 'net', 'act', 'W', 'b', 'ded_net', 'ded_act', 'ded_W', 'ded_b']

        # Take a self-snapshot against a given input batch
        if snapshot:
            self._snapshot(scope, test_dict, test_result, inp_vects, inp_names)

        # Stdout
        testName = test.name.split(sep='_')[0].split(sep=':')[0]
        if self._training:  # . . . During training
            self._inPrint('[{}] Test after epoch {}:'.format(self.name, self.counter))
            self._inPrint('[{}] Error tensor |{}| = {}'.format(self.name, testName, test_result))
        else:  # . . . . . . . . . . Before training
            self._inPrint('[{}] Initial test...'.format(self.name))
            self._inPrint('[{}] Error tensor [{}] = {}'.format(self.name, testName, test_result))
        self._last_test = self.counter
        return test_result

    def _snapshot(self, attributes, batch, test_measure, inp_vects, inp_names):
        t = []
        try:
            with open(self.logpath + '/mpl_data/snapshot_log.pkl', 'rb') as file:
                snap = pickle.load(file)
            snap['inp_vects'].append(inp_vects)
            snap['epochs'] = np.append(snap['epochs'], [self.counter], axis=0)
            snap['error'] = np.append(snap['error'], [test_measure], axis=0)
            for l in self.model['network']:
                vals = self.sess.run(self._fetch(l, attributes), feed_dict=batch)
                if l.layer_type == 'output':
                    t = ['t']
                    vals.append(batch[self.model['labels']])
                state = zip(attributes + t, vals)
                snap[l.layer_name].append(state)
            pickle.dump(snap, open(self.logpath + '/mpl_data/snapshot_log.pkl', 'wb'))
        except FileNotFoundError:
            new_snap = collections.OrderedDict({'epochs': np.array([self.counter], dtype=int),
                                                'error': np.array([test_measure], dtype=float),
                                                'attributes': attributes,
                                                'inp_vects': [inp_vects],
                                                'inp_names': inp_names})
            for l in self.model['network']:
                log = logger.LayerLog(l)
                vals = self.sess.run(self._fetch(l, attributes), feed_dict=batch)
                if l.layer_type == 'output':
                    t = ['t']
                    vals.append(batch[self.model['labels']])
                state = zip(attributes + t, vals)
                log.append(state)
                new_snap[l.layer_name] = log
            pickle.dump(new_snap, open(self.logpath + '/mpl_data/snapshot_log.pkl', 'wb'))

    def _feed_dict(self, dataset, **kwargs):
        # Fill a feed dictionary with the actual set of images and labels
        # for current training step.
        # Takes attribute self.dataset as a default data set
        try:
            batch_xs, batch_ys = dataset.next_batch(kwargs['batch_size'])
        except KeyError:
            batch_xs, batch_ys = dataset.big_batch()
        feed_dict = {}
        start = 0
        end = 0
        for inp_placeholder in self.model['images']:
            end += int(inp_placeholder.get_shape()[1])
            feed_dict[inp_placeholder] = batch_xs[:, start:end]
            start += end
        feed_dict[self.model['labels']] = batch_ys
        return feed_dict, batch_xs

    def _fetch(self, layer, scope):
        # Takes a layer object and returns a list of requested attributes
        basket = []
        for attribute in scope:
            basket.append(getattr(layer, attribute))
        return basket

    def _inBar(self, i, l):
        if self._interactive:
            printProgress(i, l, prefix='[{}] Training:'.format(self.name), barLength=15)

    def _inPrint(self, s):
        if self._interactive: print(s)

    def _checkLastTest(self):
        if self._last_test == self.counter:
            return False
        else:
            return self._settings['scope']

    def off(self):
        self.sess.close()