import numpy as np
import pickle

np.set_printoptions(precision=3, suppress=True, linewidth=200)

with open('network_snapshot.pkl', 'rb') as opened_file:
    old_snap = pickle.load(opened_file)

print('Error:')
print(old_snap['error'], end='\n\n')

for keys,subdicts in old_snap.items():
    if keys != 'error':
        print('>>> '+keys+':')
        for k, v in subdicts.items():
            print('    '+k+':')
            print(v)
