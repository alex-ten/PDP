import numpy as np
import pickle

path_to_raw = input('Path to raw letfets -> ')
FtoL_weights = np.zeros((26, 28))

with open(path_to_raw, 'r') as letfet_file:
    for row, line in enumerate(letfet_file):
        for feature, col in zip([int(x) for x in line.split()], list(range(0,28,2))):
            if feature == 1:
                FtoL_weights[row, col + 1] = 1
            else:
                FtoL_weights[row, col] = 1

print('Converted raw letfets to FtoL_weights, a (26,28) numpy array. Pickle it?')
x = input('[y/n/path] -> ')

if x == 'n':
    pass
elif x == 'y':
    with open('FtoL_weights.pkl', 'wb') as new_file:
        pickle.dump(FtoL_weights, new_file)
else:
    with open(x+'/FtoL_weights.pkl', 'wb') as new_file:
        pickle.dump(FtoL_weights, new_file)