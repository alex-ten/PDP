import numpy as np
import pickle

def intprint(x):
    # Prettily print x and shape of x
    print('shape {}'.format(np.shape(x)),'\n', x.astype(int))


def softmax(x):
    # Compute softmax values for each sets of scores in x
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def new_input(letter_ind, feature_map, batch_size = 1):
    # Given feature_map that stores feature vector representations in its rows
    # RETURN an input letter (indexed 0-25) to the model with batch_size number of
    # copies. Input must be a column vector, so the resulting new_input(args) will be
    # a 2-D array with multiple copies of a single column vector
    return np.tile(feature_map[letter_ind], [batch_size, 1]).T


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

weights_path = 'weights/'

with open(weights_path + 'FtoL_weights.pkl', 'rb') as f:
    FtoL_weights = pickle.load(f)
    features = FtoL_weights

with open(weights_path + 'L0toW_weights.pkl', 'rb') as f:
    L0toW_weights = pickle.load(f)
    WtoL0_weights = L0toW_weights.T

with open(weights_path + 'L1toW_weights.pkl', 'rb') as f:
    L1toW_weights = pickle.load(f)
    WtoL1_weights = L0toW_weights.T

with open(weights_path + 'L2toW_weights.pkl', 'rb') as f:
    L2toW_weights = pickle.load(f)
    WtoL2_weights = L0toW_weights.T

# for feature, present in zip(list(range(14)), np.reshape(np.sum(FtoL_weights, axis=0),[14,2]).astype(int)):
#     print('F{}: +{}, -{}'.format(feature,present[0],present[1]))

batch_size = 4000

prev_word = np.zeros([batch_size,36]).T

# For each element of the batch this is a vector of length NWORDS

x0 = new_input(22, features, batch_size)
x1 = new_input(14, features, batch_size)
x2 = new_input(4, features, batch_size)

L0, L0_mean = [], []
L1, L1_mean = [], []
L2, L2_mean = [], []
word, word_mean = [], []

timesteps = 20

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

print(np.around(L0_mean[-1],2))
print(np.around(L1_mean[-1],2))
print(np.around(L2_mean[-1],2))
print()
print(np.around(word_mean[-1],2))



# Running a test of the network would be done on a single specified input pattern that the user would create or choose from a list.
#
# When the test runs a graph would be shown, with four panels, one for each of L0mean, L1mean, L2mean, and Wordmean.
# The graph will show how these values change across time for different words.
#
# Finally, we will need a to create a set of pre-defined input patterns for particular test cases, as well as a
# function that could ‘generate input’ when given a sequence of three letters, as well.
# The user should then be able to manually alter the features to create any desired input pattern.
