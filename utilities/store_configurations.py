

def store(hyperparams, path):
    with open(path + '/sess_configurations.txt', 'w') as f:
        for key, val in hyperparams.items():
            if key=='Error:':
                f.write('{0:18} {1}\n'.format(key, str(val).split(' ')[1]))
            else:
                f.write('{0:18} {1}\n'.format(key, str(val)))