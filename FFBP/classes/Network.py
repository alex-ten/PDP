import collections
import pickle
import time
import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from FFBP.classes.Logger import Logger, LayerLog
from FFBP.visualization.NetworkData import NetworkData
from FFBP.visualization.VisErrorApp import VisErrorApp
from FFBP.visualization.VisLayersApp import VisLayersApp
from utilities.init_rest import init_rest
from utilities.printProgress import printProgress
from utilities.restore_params import restore_xor


class Network(object):
    def __init__(self, model, name='NN'):
        self.name = name
        self.model = model
        self.sess = tf.InteractiveSession()
        self.graph = tf.get_default_graph()
        self.logger = Logger()
        self.counter = 0
        self.train_set = None
        self.test_set = None
        self._loss = None
        self._opt = None
        self.settings = {}
        self._lossHistory = []
        self._last_test = None
        self._interactive = False
        self._training = False
        self._terminate = False
        self._errVisApp = None
        self._layVisApp = None
        self._vis_settings = {'ppc': 30, 'dpi': 96}
        self._global_step = tf.Variable(0, name='global_step', trainable=False)

    def initconfig(self,
                   loss,
                   train_batch_size,
                   learning_rate = 0.5,
                   momentum = 0.9,
                   permute = False,
                   ecrit = 0.01,
                   test_func = None,
                   test_scope = 'all'):
        self.config(loss, train_batch_size, learning_rate,
                    momentum, permute, ecrit, test_func, wrange)
        self.init_weights()

    def init_weights(self):
        # Initialize weights and biases
        Wb_vars = self.graph.get_collection('Wb')
        init_Wb_vars = tf.initialize_variables(Wb_vars)
        self.sess.run(init_Wb_vars)

    def config(self,
               loss,
               train_batch_size,
               learning_rate = 0.5,
               momentum = 0.9,
               permute = False,
               ecrit = 0.01,
               test_func = None,
               wrange = 0):
        self._loss = loss(self.model['labels'], self.model['network'][-1].act)
        self._opt = tf.train.MomentumOptimizer(learning_rate, momentum)
        self.settings['loss_func'] = loss
        self.settings['batch_size'] = train_batch_size
        self.settings['lrate'] = learning_rate
        self.settings['mrate'] = momentum
        self.settings['permute'] = permute
        self.settings['ecrit'] = ecrit
        self.settings['test_func'] = test_func
        self.settings['opt_task'] = self._opt.minimize(self._loss, global_step=self._global_step)
        self.settings['saver'] = tf.train.Saver(max_to_keep = 0)
        for l in self.model['network']:
            # When run in current session tf.gradients returns a list of numpy arrays with
            # batch_size number of rows and Layer.size number of columns.
            l.W.assign(tf.random_uniform(
                [l.sender_size, l.size],
                minval = wrange[0],
                maxval = wrange[1],
                dtype = tf.float32,
                seed = wrange[2] if len(wrange)>2 else None).eval()).eval()
            l.b.assign(tf.random_uniform(
                [1, l.size],
                minval = wrange[0],
                maxval = wrange[1],
                dtype = tf.float32,
                seed = wrange[2] if len(wrange)>2 else None).eval()).eval()
            l.ded_net = tf.gradients(self._loss, l.net)
            l.ded_act = tf.gradients(self._loss, l.act)
            l.ded_W = tf.gradients(self._loss, l.W)
            l.ded_b = tf.gradients(self._loss, l.b)
        init = init_rest()
        self.sess.run(init)

    def restore(self, path, xor=True):
        # STILL UNDER DEVELOPMENT
        # todo generalize this methods to enable restore of any set of variables
        if xor: restore_xor(path, model=self.model)

    def tnt(self, max_epochs, test_freq, checkpoints = False, **kwargs):
        max_epochs = self.counter + max_epochs
        if 'train_set' in kwargs: train_set = kwargs['train_set']
        else: train_set = self.train_set
        if 'test_set' in kwargs: test_set = kwargs['test_set']
        else: test_set = self.test_set
        print('[{}] Now in train and test mode...'.format(self.name))
        while self.counter < max_epochs:
            result = self._test(test_set,
                                evalfunc=self.settings['test_func'],
                                snapshot=self._checkLastTest(),
                                checkpoint = checkpoints)
            print('[{}] epoch {}: {}'.format(self.name, self.counter, result))
            self._train(test_freq, train_set, self.settings['batch_size'], ecrit=self.settings['ecrit'])
            if self._terminate or self.counter == max_epochs:
                result = self._test(test_set,
                                    evalfunc=self.settings['test_func'],
                                    snapshot=self._checkLastTest(),
                                    checkpoint = checkpoints)
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
                    self.settings['batch_size'],
                    self.settings['ecrit'],
                    ckpt_freq)
        if vis: self.showerr(
            str(self.settings['test_func']).split(' ')[1])
        self._interactive = False

    def test(self, vis = False):
        self._interactive = True
        self._test(dataset = self.test_set,
                   evalfunc = self.settings['test_func'],
                   snapshot = self._checkLastTest())
        if vis:
            try:
                self.shownet()
            except tk._tkinter.TclError:
                self._layVisApp = None
                self.shownet()
        self._interactive = False

    def showerr(self, error_name ='error'):
        if self.counter == 0:
            print('[{}] Run some training before viewing error.'.format(self.name))
            return
        if self._errVisApp is None:
            root1 = tk.Tk()
            root1.title('FFBPlog_{}'.format(self.logger.sess_index))
            figure = plt.figure(1, facecolor='w', dpi=self._vis_settings['dpi'])
            self._errVisApp = VisErrorApp(root1,
                                          figure,
                                          self._lossHistory,
                                          'epoch',
                                          error_name,
                                          error_name)
        if self.counter > 0:
            self._errVisApp.catch_up(self._lossHistory)

    def shownet(self):
        snap = NetworkData(self.logger.child_path + '/snap.pkl')
        if self._layVisApp is None:
            root2 = tk.Tk()
            self._layVisApp = VisLayersApp(root2, snap, self._vis_settings['ppc'])
        else:
            self._layVisApp.catch_up(snap)

    def restart(self):
        # STILL UNDER DEVELOPMENT
        print('[{}] Restarting network ...'.format(self.name))
        self.sess = tf.InteractiveSession()
        self.graph = tf.get_default_graph()
        self.counter = 0
        self._lossHistory = []
        self._interactive = False
        self._training = False
        self._terminate = False
        if self._errVisApp is not None: self._errVisApp.destroy()
        if self._layVisApp is not None: self._layVisApp.destroy()
        self._errVisApp = None
        self._layVisApp = None
        self._global_step = tf.Variable(0, name='global_step', trainable=False)
        print('[{}] Restarting complete.'.format(self.name))

    def _train(self, num_epochs, dataset, batch_size, ecrit=0.01, permute = False):
        if not self._training: self._training = True
        if self._terminate:
            return
        else:
            if 'permute' in self.settings:
                permute = self.settings['permute']
            t0 = self.counter
            t1 = t0 + num_epochs
            start = time.time()
            self._inBar(t0, t1)
            for step in range(t0, t1):
                if permute: dataset.permute()
                if t1 > 1:
                    self._inBar(step, t1 - 1)
                else:
                    self._inBar(1, 1)

                epoch_loss = 0
                for j in range(int(dataset._num_examples/batch_size)):
                    train_dict, _ = self._feed_dict(dataset, batch_size=batch_size)
                    _, loss_val = self.sess.run([self.settings['opt_task'], self._loss],
                                                feed_dict=train_dict)
                    epoch_loss += loss_val

                # Collect stats (note that loss is measured before the gradients are applied):
                self._lossHistory.append(epoch_loss)

                if epoch_loss < ecrit:
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

    def _test(self, dataset, evalfunc, snapshot=False, checkpoint=False):
        test_dict, inp_vects = self._feed_dict(dataset)
        inp_names = dataset.names
        test = evalfunc(self.model['labels'], self.model['network'][-1].act)

        # Evaluate test measure
        test_result = test.eval(feed_dict=test_dict)
        scope = ['inp', 'net', 'act', 'W', 'b', 'ded_net', 'ded_act', 'ded_W', 'ded_b']

        # Take a self-snapshot against a given input batch
        if snapshot:
            self._snapshot(scope, test_dict, test_result, inp_vects, inp_names)

        # Save a checkpoint periodically
        if checkpoint:
            self.settings['saver'].save(self.sess,
                                        self.logger.child_path + '/weights',
                                        global_step=self.counter)

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
            with open(self.logger.child_path + '/snap.pkl', 'rb') as file:
                snap = pickle.load(file)
            snap['inp_vects'].append(inp_vects)
            snap['epochs'] = np.append(snap['epochs'], [self.counter], axis=0)
            snap['error'] = np.append(snap['error'], [test_measure], axis=0)
            # TO-DO: FIGURE OUT AN ELEGANT WAY TO DO THE APPENDING ================
            snap['hyperparams'].append([self.settings['lrate'],self.settings['mrate'],self.settings['loss_func'],self.settings['batch_size'], self.settings['permute']])
            for l in self.model['network']:
                vals = self.sess.run(self._fetch(l, attributes), feed_dict=batch)
                if l.layer_type == 'output':
                    t = ['t']
                    vals.append(batch[self.model['labels']])
                state = zip(attributes + t, vals)
                snap[l.layer_name].append(state)
            pickle.dump(snap, open(self.logger.child_path + '/snap.pkl', 'wb'))
        except FileNotFoundError:
            new_snap = collections.OrderedDict({'epochs': np.array([self.counter], dtype=int),
                                                'error': np.array([test_measure], dtype=float),
                                                'attributes': attributes,
                                                'inp_vects': [inp_vects],
                                                'inp_names': inp_names,
                                                'hyperparams': [[self.settings['lrate'],
                                                                self.settings['mrate'],
                                                                self.settings['loss_func'],
                                                                self.settings['batch_size'],
                                                                self.settings['permute']]],
                                               'sess_index': self.logger.sess_index})
            for l in self.model['network']:
                log = LayerLog(l)
                vals = self.sess.run(self._fetch(l, attributes), feed_dict=batch)
                if l.layer_type == 'output':
                    t = ['t']
                    vals.append(batch[self.model['labels']])
                state = zip(attributes + t, vals)
                log.append(state)
                new_snap[l.layer_name] = log
            pickle.dump(new_snap, open(self.logger.child_path + '/snap.pkl', 'wb'))

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
            return True

    def off(self):
        self.sess.close()