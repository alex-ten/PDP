def store_hyper_params(hyperparams, path):
    with open(path+'/session_hyperparmeters.txt', 'w') as f:
        for key, val in hyperparams.items():
            f.write('{0:18} {1}\n'.format(key, str(val)))