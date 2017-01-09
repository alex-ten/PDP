import matplotlib; matplotlib.use('TkAgg')
import os
import pickle
import argparse
import numpy as np
import tkinter as tk
from MIA.visualization.MIA_Viewer import MIA_Viewer

from utilities.logger import logdir


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
    parser.add_argument('-olw', '--oletwrd', help='set odds correct letter given word', action='store', type=float)
    parser.add_argument('-ofl', '--ofetlet', help='set odds correct feature given letter', action='store', type=float)
    parser.add_argument('-t', '--timesteps', help='number of timesteps to run the model', type=int)
    parser.add_argument('-nw', '--no_word', help='turn off top-down word-to-letter signal', action='store_true')
    group.add_argument('-s', '--save', help='save choices at each time step', action='store_true')
    group.add_argument('-v', '--visualize', help='follow up the simulation with visualization', action='store_true')
    group.add_argument('-sv', '--save_and_visualize', help='save choices at each time step and follow up with visualization', action='store_true')
    args = parser.parse_args()

    batch_size = 1000
    timesteps = 20
    OLgivenW = 10
    OFgivenL = 10
    numwords = 36

    if args.batch_size is not None: batch_size = args.batch_size
    if args.oletwrd is not None: OLgivenW = args.scalar1
    if args.ofetlet is not None: OFgivenL = args.scalar2
    if args.timesteps is not None: timesteps = args.timesteps


    WtoLScaleFactor = np.log(OLgivenW)
    LtoFScaleFactor = np.log(OFgivenL)

    weights_path = os.getcwd()+'/MIA/weights/'
    features_path = os.getcwd()+'/MIA/raw/'

    with open(weights_path + 'FtoL_weights.pkl', 'rb') as f:
        FtoL_weights = pickle.load(f)

    with open(weights_path + 'L0toW_weights.pkl', 'rb') as f:
        L0toW_weights = pickle.load(f)

    with open(weights_path + 'L1toW_weights.pkl', 'rb') as f:
        L1toW_weights = pickle.load(f)

    with open(weights_path + 'L2toW_weights.pkl', 'rb') as f:
        L2toW_weights = pickle.load(f)

    with open(features_path + 'features.pkl', 'rb') as f:
        features = pickle.load(f)

    FtoL_weights = FtoL_weights * LtoFScaleFactor
    L0toW_weights = L0toW_weights * WtoLScaleFactor
    L1toW_weights = L1toW_weights * WtoLScaleFactor
    L2toW_weights = L2toW_weights * WtoLScaleFactor
    WtoL0_weights = L0toW_weights.T
    WtoL1_weights = L1toW_weights.T #corrected by jlm
    WtoL2_weights = L2toW_weights.T

    if args.no_word:
        WtoL0_weights *= 0
        WtoL1_weights *= 0
        WtoL2_weights *= 0

    if args.save or args.save_and_visualize:
        logdir_path = logdir(TF=False)

    proceed = True
    first = True

    root = tk.Tk()
    root.state('withdrawn')


    while proceed:
        prev_word = np.zeros([batch_size,36]).T

        # For each element of the batch this is a vector of length NWORDS
        print('[MIA network] Starting new simulation...')
        inp = input('[MIA network] Enter input string: ')
        x0, x1, x2 = new_input(inp, features, batch_size)

        L0, L0_mean = [], []
        L1, L1_mean = [], []
        L2, L2_mean = [], []
        word, word_mean = [], []
        log = {}

        #conpute marginals from sums of state goodnesses
        # bottom up signal to letters
        # Intending to use only the first copy of xI
        # to get a single copy of the feature vector
        gL0 = np.dot(FtoL_weights, x0.T[0])
        gL1 = np.dot(FtoL_weights, x1.T[0])
        gL2 = np.dot(FtoL_weights, x2.T[0])
        tEGW = np.zeros(numwords) #same shape as word but just one column
        tEGL0 = np.zeros(26) #same shape as L0 but just one column
        tEGL1 = np.zeros(26)
        tEGL2 = np.zeros(26)

        for w in range(numwords):
          for l0 in range(26):
            for l1 in range(26):
              for l2 in range(26):
                gw = np.log(1/numwords) + L0toW_weights[w,l0] + L1toW_weights[w,l1] + L2toW_weights[w,l2];
                eGS = np.exp(gw + gL0[l0] + gL1[l1] + gL2[l2])
                tEGL0[l0] += eGS
                tEGL1[l1] += eGS
                tEGL2[l2] += eGS
                tEGW[w] += eGS

        pEGL0 = tEGL0/np.sum(tEGL0)
        pEGL1 = tEGL1/np.sum(tEGL1)
        pEGL2 = tEGL2/np.sum(tEGL2)
        pEGW = tEGW/np.sum(tEGW)
        rprint(pEGL0)
        rprint(pEGL1)
        rprint(pEGL2)
        rprint(pEGW)

        log['input'] = (x0.T[0], x1.T[0], x2.T[0])
        log['L0_marginal'] = pEGL0
        log['L1_marginal'] = pEGL1
        log['L2_marginal'] = pEGL2
        log['word_marginal'] = pEGW

        # end of computing marginals

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
            # L0.append(rci_L0)
            # L1.append(rci_L1)
            # L2.append(rci_L2)
            # word.append(rci_W)

            # update prev_word
            prev_word = one_hot(36, rci_W)


        # Then the next computations are getting average values across all of the elements of the batch:
            # Note that L1_mean[t] is a vector of length 26 showing for each letter the proportion of cases in which
            # L1[t] was 1 for that letter in the first letter position

            L0_mean.append(np.mean(rcv_L0,1))
            L1_mean.append(np.mean(rcv_L1,1))
            L2_mean.append(np.mean(rcv_L2,1))

            word_mean.append(np.mean(prev_word,1))

        log['L0_mean'] = L0_mean
        log['L1_mean'] = L1_mean
        log['L2_mean'] = L2_mean
        log['word_mean'] = word_mean
        log['input'] = (x0.T[0], x1.T[0], x2.T[0])

        if args.save:
            with open(logdir_path + '/mpl_data/log_dict-{}.pkl'.format(inp), 'wb') as new_file:
                pickle.dump(log, new_file)

        if args.visualize:
            if first:
                VisApp = MIA_Viewer(root, log)
                first = False
            else: VisApp.plot_new_data(log)


        if args.save_and_visualize:
            with open(logdir_path + '/mpl_data/log_dict-{}.pkl'.format(inp), 'wb') as new_file:
                pickle.dump(log, new_file)
            if first:
                VisApp = MIA_Viewer(root, log)
                first = False
            else: VisApp.plot_new_data(log)

        print('[MIA network] Simulation terminated.')
        print('[MIA network] Would you like to run another simulation?')
        run = input('[y/n] -> ')
        if run == 'n':
            proceed = False
            print('[MIA network] Session terminated.')
        else:
            continue



if __name__=='__main__': main()