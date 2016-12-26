import matplotlib; matplotlib.use('Agg')
import numpy as np
import pickle
from utilities.logger import logdir
import tkinter as tk
from classes.MIA_Viewer import MIA_Viewer
import math
import os
import argparse

def rprint(x):
    print(np.around(x,2))


def intprint(x):
    # Prettily print x and shape of x
    print('shape {}'.format(np.shape(x)),'\n', x.astype(int))


def softmax(x):
    # Compute softmax values for each sets of scores in x
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def new_input(s, features, batch_size = 1):
    # special symbols
    underbar, hash, at, questionmark = features[0], features[0], features[1], features[4]
    underbar = underbar * 0
    hash[[4,5]] = 0
    at[[16,17]] = 0
    questionmark[[6,7,8,9,12,13]] = 0
    special = np.array([underbar, hash, at, questionmark])
    features = np.row_stack([features,special])
    alphabet = list('abcdefghijklmnopqrstuvwxyz_#@?')
    xs = list(s.lower())
    inds = [alphabet.index(x) for x in xs]
    return [np.tile(features[ind], [batch_size, 1]).T for ind in inds]


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

# ============================= MIA Model =============================
# The goal is to infer:
#   >> the identity of the word and
#   >> the identities of the four letters
# ... that generated the features that reach the input to the model
# in a trial of a perception experiment using displays containing
# features in four letter positions
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('-bs', '--batch_size', help='size of the input batch to be processed in parallel', type=int)
    parser.add_argument('-sc1', '--scalar1', help='scale bottom-up weights', action='store', type=float)
    parser.add_argument('-sc2', '--scalar2', help='scale top-down weights', action='store', type=float)
    parser.add_argument('-in', '--input', help='input to the model', action='store', type=str)
    parser.add_argument('-t', '--timesteps', help='number of timesteps to run the model', type=int)
    group.add_argument('-s', '--save', help='save choices at each time step', action='store_true')
    group.add_argument('-v', '--visualize', help='follow up the simulation with visualization', action='store_true')
    group.add_argument('-sv', '--save_and_visualize', help='save choices at each time step and follow up with visualization', action='store_true')


    args = parser.parse_args()

    batch_size = 1000
    PLgivenW = .9
    PFgivenL = .9
    if args.batch_size is not None: batch_size = args.batch_size
    if args.scalar1 is not None: PLgivenW = args.scalar1
    if args.scalar2 is not None: PFgivenL = args.scalar2


    WtoLScaleFactor = math.log(PLgivenW / ((1 - PLgivenW) / 25))
    LtoFScaleFactor = math.log(PFgivenL /(1 - PFgivenL))

    weights_path = os.getcwd()+'/MIA/weights/'

    with open(weights_path + 'FtoL_weights.pkl', 'rb') as f:
        FtoL_weights = pickle.load(f)
        features = FtoL_weights

    with open(weights_path + 'L0toW_weights.pkl', 'rb') as f:
        L0toW_weights = pickle.load(f) * WtoLScaleFactor

    with open(weights_path + 'L1toW_weights.pkl', 'rb') as f:
        L1toW_weights = pickle.load(f) * WtoLScaleFactor

    with open(weights_path + 'L2toW_weights.pkl', 'rb') as f:
        L2toW_weights = pickle.load(f) * WtoLScaleFactor



    FtoL_weights = FtoL_weights * LtoFScaleFactor
    L0toW_weights = L0toW_weights * WtoLScaleFactor
    WtoL0_weights = L0toW_weights.T
    L1toW_weights = L1toW_weights * WtoLScaleFactor
    WtoL1_weights = L0toW_weights.T
    L2toW_weights = L2toW_weights * WtoLScaleFactor
    WtoL2_weights = L0toW_weights.T

    prev_word = np.zeros([batch_size,36]).T

    # For each element of the batch this is a vector of length NWORDS
    x0, x1, x2 = new_input('age', features, batch_size)
    if args.input is not None:
        x0, x1, x2 = new_input(args.input, features, batch_size)

    L0, L0_mean = [], []
    L1, L1_mean = [], []
    L2, L2_mean = [], []
    word, word_mean = [], []

    timesteps = 20
    if args.timesteps is not None: timesteps = args.timesteps

    for t in range(timesteps):
        # bottom up signal
        bus_L0 = np.dot(FtoL_weights, x0)
        bus_L1 = np.dot(FtoL_weights, x1)
        bus_L2 = np.dot(FtoL_weights, x2)

        # top down signal
        tds_L0 = np.dot(WtoL0_weights, prev_word)
        tds_L1 = np.dot(WtoL1_weights, prev_word)
        tds_L2 = np.dot(WtoL2_weights, prev_word)

        # logits
        logit_L0 = bus_L0 + tds_L0
        logit_L1 = bus_L1 + tds_L1
        logit_L2 = bus_L2 + tds_L2
        # rprint(logit_L0.T)
        # rprint(logit_L1.T)
        # rprint(logit_L2.T)

        # probabilities
        probs_L0 = softmax(logit_L0)
        probs_L1 = softmax(logit_L1)
        probs_L2 = softmax(logit_L2)
        # rprint(probs_L0.T)
        # rprint(probs_L1.T)
        # rprint(probs_L2.T)

        # random choice inds
        rci_L0 = choose_one(probs_L0)
        rci_L1 = choose_one(probs_L1)
        rci_L2 = choose_one(probs_L2)
        # print(rci_L0)
        # print(rci_L1)
        # print(rci_L2)

        # random choice vectors
        rcv_L0 = one_hot(26, rci_L0)
        rcv_L1 = one_hot(26, rci_L1)
        rcv_L2 = one_hot(26, rci_L2)

        # another bottom up signal
        abus_L0 = np.dot(L0toW_weights, rcv_L0)
        abus_L1 = np.dot(L1toW_weights, rcv_L1)
        abus_L2 = np.dot(L2toW_weights, rcv_L2)

        # word logit
        logit_W =  abus_L0 + abus_L1 + abus_L2

        # word probability
        prob_W = softmax(logit_W)

        # word random choice ind
        rci_W = choose_one(prob_W)

        # store inferences
        L0.append(rci_L0)
        L1.append(rci_L1)
        L2.append(rci_L2)
        word.append(rci_W)

        # update prev_word
        prev_word = one_hot(36, rci_W)


    # Then the next computations are getting average values across all of the elements of the batch:
        # Note that L1_mean[t] is a vector of length 26 showing for each letter the proportion of cases in which
        # L1[t] was 1 for that letter in the first letter position

        L0_mean.append(np.mean(rcv_L0,1))
        L1_mean.append(np.mean(rcv_L1,1))
        L2_mean.append(np.mean(rcv_L2,1))

        word_mean.append(np.mean(prev_word,1))

    log = {'L0_mean': L0_mean,
           'L1_mean': L1_mean,
           'L2_mean': L2_mean,
           'word_mean': word_mean}
    logdir_path = logdir(TF=False)

    if args.save:
        with open(logdir_path + '/mpl_data/log_dict.pkl', 'wb') as new_file:
            pickle.dump(log, new_file)

    if args.visualize:
        root = tk.Tk()
        VisApp = MIA_Viewer(root, log)

    if args.save_and_visualize:
        with open(logdir_path + '/mpl_data/log_dict.pkl', 'wb') as new_file:
            pickle.dump(log, new_file)
        root = tk.Tk()
        VisApp = MIA_Viewer(root, log)


if __name__=='__main__': main()