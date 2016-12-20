import tensorflow as tf
import numpy as np
import pickle
from choose1 import choose1


# ============================= MIA Model =============================
# The goal is to infer:
#   (1)   the identity of the word and
#   (2-5) the identities of the four letters
# ... that generated the features that reach the input to the model
# in a trial of a perception experiment using displays containing
# features in four letter positions
# ======================================================================

weights_path = 'weights/'

with open(weights_path + 'FtoL_weights.pkl', 'rb') as f:
    FtoL_weights = pickle.load(f)
    inps = FtoL_weights
    FtoL_weights = tf.Variable(FtoL_weights, trainable=False, dtype=tf.float32)

with open(weights_path + 'L0toW_weights.pkl', 'rb') as f:
    L0toW_weights = pickle.load(f)
    temp = L0toW_weights
    L0toW_weights = tf.Variable(temp, trainable=False, dtype=tf.float32)
    WtoL0_weights = tf.Variable(temp.T, trainable=False, dtype=tf.float32)

with open(weights_path + 'L1toW_weights.pkl', 'rb') as f:
    L1toW_weights = pickle.load(f)
    temp = L1toW_weights
    L1toW_weights = tf.Variable(temp, trainable=False, dtype=tf.float32)
    WtoL1_weights = tf.Variable(temp.T, trainable=False, dtype=tf.float32)

with open(weights_path + 'L2toW_weights.pkl', 'rb') as f:
    L2toW_weights = pickle.load(f)
    temp = L2toW_weights
    L2toW_weights = tf.Variable(temp, trainable=False, dtype=tf.float32)
    WtoL2_weights = tf.Variable(temp.T, trainable=False, dtype=tf.float32)

# for feature, present in zip(list(range(14)), np.reshape(np.sum(FtoL_weights, axis=0),[14,2]).astype(int)):
#     print('F{}: +{}, -{}'.format(feature,present[0],present[1]))

prev_word = tf.zeros((36,1), dtype=tf.float32)

# For each element of the batch this is a vector of length NWORDS

inp_0 = tf.constant(np.reshape(inps[0],(28,1)), dtype=tf.float32)
inp_1 = tf.constant(np.reshape(inps[8],(28,1)), dtype=tf.float32)
inp_2 = tf.constant(np.reshape(inps[3],(28,1)), dtype=tf.float32)

L0 = []
L1 = []
L2 = []
word = []

timesteps = 20

for t in range(timesteps):
    L0.append(
        choose1(
            tf.nn.softmax(
                tf.matmul(FtoL_weights, inp_0) +
                tf.matmul(WtoL0_weights, prev_word)
            )
        )
    )

    L1.append(choose1(tf.nn.softmax(tf.matmul(FtoL_weights, inp_1) + tf.matmul(WtoL1_weights, prev_word))))
    L2.append(choose1(tf.nn.softmax(tf.matmul(FtoL_weights, inp_2) + tf.matmul(WtoL2_weights, prev_word))))

    word.append(
        choose1(
            tf.nn.softmax(tf.matmul(L0toW_weights, L0[t]) +
                          tf.matmul(L1toW_weights, L1[t]) +
                          tf.matmul(L2toW_weights, L2[t])
            )
        )
    )

    prev_word = word

# Then the next computations are getting average values across all of the elements of the batch:

    L0_mean = tf.reduce_mean(L0[t])
    L1_mean = tf.reduce_mean(L1[t])
    L2_mean = tf.reduce_mean(L2[t])
    word_mean = tf.reduce_mean(word[t])

with tf.InteractiveSession() as sess:
    print(L0[0].eval())
# Note that L1mean(t) is a vector of length 26 showing for each letter the proportion of cases in which L1(t) was 1 for that letter in the first letter position.
#
# Running a test of the network would be done on a single specified input pattern that the user would create or choose from a list.
#
# When the test runs a graph would be shown, with four panels, one for each of L0mean, L1mean, L2mean, and Wordmean.  The graph will show how these values change across time for different words.
#
# Finally, we will need a to create a set of pre-defined input patterns for particular test cases, as well as a function that could ‘generate input’ when given a sequence of three letters, as well.  The user should then be able to manually alter the features to create any desired input pattern.
#
# OK, this should give you enough to work on for now.  Thanks for your effort on this!