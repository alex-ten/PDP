import numpy as np
import datetime as dt
import os

def logdir():
    # Create logdir directory if it doesn't exist
    logdir_dir = os.getcwd() + '/logdir'
    try:
        os.mkdir(logdir_dir)
        # Name sess directory according to current date-time
        sess_name = 'Sess_' + dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        dir_path = logdir_dir + '/' + sess_name
        os.mkdir(dir_path)
        os.mkdir(dir_path + '/tf_params')
        os.mkdir(dir_path + '/mpl_data')
    except FileExistsError:
        sess_name = 'Sess_'+dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        dir_path = logdir_dir + '/' + sess_name
        os.mkdir(dir_path)
        os.mkdir(dir_path + '/tf_params')
        os.mkdir(dir_path + '/mpl_data')
    return dir_path

def unroll(ar, index):
    if type(ar) is list: ar = ar[0]
    strip1D = np.insert(ar.flatten(), 0, [index] + list(ar.shape))
    return np.reshape(strip1D, (1,strip1D.size))

def rollup(strip):
    pass

def append_snapshot(old, new):
    for KEY in old.keys():
        if KEY == 'error':
            old[KEY] = np.append(old[KEY], new[KEY], axis=0)
        else:
            for key in old[KEY].keys():
                old[KEY][key] = np.append(old[KEY][key], new[KEY][key], axis=0)
    return old
