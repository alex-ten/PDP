import os
import pickle
import tkinter as tk

import numpy as np

from MIA.classes.Logger import Logger
from MIA.classes.MIAInput import MIAInput
from MIA.classes.MIAViewer import ViewerExecutive


def softmax(x):
    # Compute softmax values for each sets of scores in x
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def one_hot(length, ind):
    # RETURN one-hot column vector(s) of a given length
    # where a nonzero element is indexed by ind
    batch_size = len(ind)
    zeros = np.zeros((length, batch_size))
    zeros[ind, np.arange(batch_size)] = 1
    return zeros


def choose_one(x):
    # RETURN the index an element of x according to the probability distribution over x
    length, batch_size = np.shape(x)
    csm_out = np.cumsum(x, axis=0)
    s = np.random.rand(1,batch_size)
    choice = s < csm_out
    return length - np.sum(choice,0)


class MIANetwork(object):
    def __init__(self, OLgivenW, OFgivenL, batch_size = 1000, timesteps = 20, top_down = True, name='MIA network'):
        weights_path = os.getcwd() + '/MIA/weights/'

        with open(weights_path + 'FtoL_weights.pkl', 'rb') as f:
            self.FtoL_weights = pickle.load(f)

        with open(weights_path + 'L0toW_weights.pkl', 'rb') as f:
            self.L0toW_weights = pickle.load(f)

        with open(weights_path + 'L1toW_weights.pkl', 'rb') as f:
            self.L1toW_weights = pickle.load(f)

        with open(weights_path + 'L2toW_weights.pkl', 'rb') as f:
            self.L2toW_weights = pickle.load(f)

        self.WtoLScaleFactor = np.log(OLgivenW)
        self.LtoFScaleFactor = np.log(OFgivenL)

        self.FtoL_weights = self.FtoL_weights * self.LtoFScaleFactor
        self.L0toW_weights = self.L0toW_weights * self.WtoLScaleFactor
        self.L1toW_weights = self.L1toW_weights * self.WtoLScaleFactor
        self.L2toW_weights = self.L2toW_weights * self.WtoLScaleFactor

        self.WtoL0_weights = self.L0toW_weights.T
        self.WtoL1_weights = self.L1toW_weights.T
        self.WtoL2_weights = self.L2toW_weights.T

        self.batchsize = batch_size
        self.timesteps = timesteps
        self.topdown = top_down
        self.name = name
        self.viewer_exec = ViewerExecutive(tk.Tk())

        self.Logger = Logger()
        self.log = None
        self._first = True
        self._sim_count = 1

    def run(self, s, mask0=None, mask1=None, mask2=None, vis=True):
        NWORDS = 36
        L0, L0_mean = [], []
        L1, L1_mean = [], []
        L2, L2_mean = [], []
        word, word_mean = [], []
        self.log = {'batch_size': self.batchsize, 'w2l': self.WtoLScaleFactor, 'l2f': self.LtoFScaleFactor}

        # For each element of the batch this is a vector of length NWORDS
        print('[{}] Starting new simulation (sim{})...'.format(self.name, self._sim_count))
        mia_inp = MIAInput(s, self.batchsize)
        for i, mask in enumerate((mask0, mask1, mask2)):
            if mask is not None: mia_inp.mask(i, mask)

        x0, x1, x2 = mia_inp()

        # Compute marginals from sums of state goodnesses
        # bottom up signal to letters
        # Intending to use only the first copy of xI
        # to get a single copy of the feature vector
        gL0 = np.dot(self.FtoL_weights, x0.T[0])
        gL1 = np.dot(self.FtoL_weights, x1.T[0])
        gL2 = np.dot(self.FtoL_weights, x2.T[0])
        tEGW = np.zeros(NWORDS)  # same shape as word but just one column
        tEGL0 = np.zeros(26)  # same shape as L0 but just one column
        tEGL1 = np.zeros(26)
        tEGL2 = np.zeros(26)


        if self.topdown:
            for w in range(NWORDS):
                for l0 in range(26):
                    for l1 in range(26):
                        for l2 in range(26):
                            gw = (np.log(1 / NWORDS) + self.L0toW_weights[w, l0]
                                  + self.L1toW_weights[w, l1] + self.L2toW_weights[w, l2])
                            eGS = np.exp(gw + gL0[l0] + gL1[l1] + gL2[l2])
                            tEGL0[l0] += eGS
                            tEGL1[l1] += eGS
                            tEGL2[l2] += eGS
                            tEGW[w] += eGS
        else:
            for l0 in range(26):
                for l1 in range(26):
                    for l2 in range(26):
                        eGS = np.exp(gL0[l0] + gL1[l1] + gL2[l2])
                        tEGL0[l0] += eGS
                        tEGL1[l1] += eGS
                        tEGL2[l2] += eGS

        pEGL0 = tEGL0 / np.sum(tEGL0)
        pEGL1 = tEGL1 / np.sum(tEGL1)
        pEGL2 = tEGL2 / np.sum(tEGL2)
        pEGW = tEGW / np.sum(tEGW)
        sumpp = np.sum(tEGW) / np.exp(3 * np.log(35) + 42 * np.log(11))

        self.log['sumpp'] = sumpp
        self.log['input'] = (x0.T[0], x1.T[0], x2.T[0])
        self.log['L0_marginal'] = pEGL0
        self.log['L1_marginal'] = pEGL1
        self.log['L2_marginal'] = pEGL2
        self.log['word_marginal'] = pEGW
        # end of computing marginals

        prev_word = np.zeros([self.batchsize, 36]).T

        for t in range(self.timesteps):
            # bottom up signal
            bus_L0 = np.dot(self.FtoL_weights, x0)
            bus_L1 = np.dot(self.FtoL_weights, x1)
            bus_L2 = np.dot(self.FtoL_weights, x2)

            # top down signal
            if self.topdown:
                tds_L0 = np.dot(self.WtoL0_weights, prev_word)
                tds_L1 = np.dot(self.WtoL1_weights, prev_word)
                tds_L2 = np.dot(self.WtoL2_weights, prev_word)
            else:
                tds_L0 = np.dot(self.WtoL0_weights * 0, prev_word)
                tds_L1 = np.dot(self.WtoL1_weights * 0, prev_word)
                tds_L2 = np.dot(self.WtoL2_weights * 0, prev_word)

            # logits
            logit_L0 = bus_L0 + tds_L0
            logit_L1 = bus_L1 + tds_L1
            logit_L2 = bus_L2 + tds_L2

            # probabilities
            probs_L0 = softmax(logit_L0)
            probs_L1 = softmax(logit_L1)
            probs_L2 = softmax(logit_L2)

            # random choice inds
            rci_L0 = choose_one(probs_L0)
            rci_L1 = choose_one(probs_L1)
            rci_L2 = choose_one(probs_L2)

            # random choice vectors
            rcv_L0 = one_hot(26, rci_L0)
            rcv_L1 = one_hot(26, rci_L1)
            rcv_L2 = one_hot(26, rci_L2)

            # another bottom up signal
            abus_L0 = np.dot(self.L0toW_weights, rcv_L0)
            abus_L1 = np.dot(self.L1toW_weights, rcv_L1)
            abus_L2 = np.dot(self.L2toW_weights, rcv_L2)

            # word logit
            logit_W = abus_L0 + abus_L1 + abus_L2

            # word probability
            prob_W = softmax(logit_W)

            # word random choice ind
            rci_W = choose_one(prob_W)

            # store inferences
            # L0.append(rci_L0)
            # L1.append(rci_L1)
            # L2.append(rci_L2)
            # word.append(rci_W)

            # update prev_word
            prev_word = one_hot(36, rci_W)

            # Then the next computations are getting average values across all of the elements of the batch:
            # Note that L1_mean[t] is a vector of length 26 showing for each letter the proportion of cases in which
            # L1[t] was 1 for that letter in the first letter position

            L0_mean.append(np.mean(rcv_L0, 1))
            L1_mean.append(np.mean(rcv_L1, 1))
            L2_mean.append(np.mean(rcv_L2, 1))

            word_mean.append(np.mean(prev_word, 1))

        self.log['L0_mean'] = L0_mean
        self.log['L1_mean'] = L1_mean
        self.log['L2_mean'] = L2_mean
        self.log['word_mean'] = word_mean

        self.Logger.may_be_make_parent()
        fullname= self.Logger.save(self.log)
        print('[{}] Saved log to \"{}\".'.format(self.name,
                                               fullname))

        if vis:
            print('[{}] Displaying visualization...'.format(self.name))
            self.viewer_exec.view(fullname)

        self._sim_count += 1
        print('[{}] Simulation terminated.'.format(self.name))

    def setw2l(self, OLgivenW):
        self.WtoLScaleFactor = np.log(OLgivenW)
        self.L0toW_weights = self.L0toW_weights * self.WtoLScaleFactor
        self.L1toW_weights = self.L1toW_weights * self.WtoLScaleFactor
        self.L2toW_weights = self.L2toW_weights * self.WtoLScaleFactor

        self.WtoL0_weights = self.L0toW_weights.T
        self.WtoL1_weights = self.L1toW_weights.T
        self.WtoL2_weights = self.L2toW_weights.T
        print('[{}] Word to Letter scale factor set to {}'.format(self.name, self.WtoLScaleFactor))

    def setl2f(self, OFgivenL):
        self.LtoFScaleFactor = np.log(OFgivenL)
        self.FtoL_weights = self.FtoL_weights * self.LtoFScaleFactor
        print('[{}] Feature to Letter scale factor set to {}'.format(self.name, self.LtoFScaleFactor))

    def settd(self, b):
        if (b == 'off' or b == False or b == 0) and self.topdown == True:
            self.topdown = False
            print('[{}] Top-down signal is now off'.format(self.name))
        if (b == 'on' or b == True or b == 1) and self.topdown == False:
            self.topdown = True
            print('[{}] Top-down signal is now on'.format(self.name))