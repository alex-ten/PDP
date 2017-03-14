import numpy as np
import code

from FFBP.visualization.NetworkData import NetworkData
from PDPATH import PDPATH

def main():
    proceed = True
    usrdir = input('[FFBP Reader] Provide user directory (if any), or press \'enter\' to use default directory: ')
    usrdir = usrdir.strip()
    while proceed:
        path = input('[FFBP Reader] Enter name of log directory OR corresponding index: ')
        try:
            ID = int(path)
            path = PDPATH('/FFBP{}/logs/FFBPlog_{}/snap.pkl'.format('/' + usrdir if len(usrdir) else '', path))
        except ValueError:
            ID = int(path.split(sep='_')[-1])
            path = PDPATH('/FFBP{}/logs/'.format('/' + usrdir if len(usrdir) else '') + path + '/snap.pkl')
        with open(path, 'rb'):
            reader = NetworkData(path)
        code.interact(local=locals())

        print('[FFBP Reader] Would you like to proceed?')
        prompt = input("[y/n] -> ")
        if prompt == 'n': proceed = False


if __name__=='__main__': main()