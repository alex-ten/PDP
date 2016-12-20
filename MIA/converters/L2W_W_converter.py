import numpy as np
import pickle

path_to_raw = input('Path to raw letfets -> ')

num_words = 36
num_letters = 26

L0toW_weights = np.zeros((num_words, num_letters))
L1toW_weights = np.zeros((num_words, num_letters))
L2toW_weights = np.zeros((num_words, num_letters))

with open(path_to_raw, 'r') as inds_file:
    for j, line, in enumerate(inds_file):
        inds = [int(x) for x in line.split()]
        L0toW_weights[j, inds[0]] = 1
        L1toW_weights[j, inds[1]] = 1
        L2toW_weights[j, inds[2]] = 1

print('Converted raw indexes to three LtoW_weights, all (36,26) numpy arrays. Pickle them?')
x = input('[y/n/path] -> ')

weights = [L0toW_weights, L1toW_weights, L2toW_weights]
if x == 'n':
    pass
elif x == 'y':
    for i, w in zip(list(range(0,3)), weights):
        with open('L{}toW_weights.pkl'.format(i), 'wb') as new_file:
            pickle.dump(w, new_file)

else:
    for i, w in zip(list(range(0,3)), weights):
        with open(x+'/L{}toW_weights.pkl'.format(i), 'wb') as new_file:
            pickle.dump(w, new_file)