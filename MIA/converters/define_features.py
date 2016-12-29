import numpy as np
import pickle

path_to_raw = input('Path to raw letfets -> ')
features = np.zeros((30, 28)) # shape of the array to be stored

with open(path_to_raw, 'r') as letfet_file:
    for row, line in enumerate(letfet_file):
        for feature, col in zip([int(x) for x in line.split()], list(range(0,28,2))):
            if feature == 1:
                features[row, col + 1] = 1
            else:
                features[row, col] = 1

features = features.astype(bool)
features[27] += features[0] & features[7]
features[28] += features[1] & features[3]
features[29] += features[4] & features[14]
features = features.astype(int)

print('Converted raw letfets to FtoL_weights, a (26,28) numpy array. Pickle it?')
x = input('[y/n/path] -> ')

if x == 'n':
    pass
elif x == 'y':
    with open('features.pkl', 'wb') as new_file:
        pickle.dump(features, new_file)
else:
    with open(x+'/features.pkl', 'wb') as new_file:
        pickle.dump(features, new_file)