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
        os.mkdir(dir_path + '/params')
        os.mkdir(dir_path + '/events')
    except FileExistsError:
        sess_name = 'Sess_'+dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        dir_path = logdir_dir + '/' + sess_name
        os.mkdir(dir_path)
        os.mkdir(dir_path + '/params')
        os.mkdir(dir_path + '/events')
    return dir_path