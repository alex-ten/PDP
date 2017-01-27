import tkinter as tk
from FFBP.visualization.VisLayersApp import VisLayersApp as VLA
import pickle
import os

class ViewerExecutive():
    def __init__(self, master):
        self.master = master
        self.master.state('withdrawn')
        self.viewers = []
        self._intcount = 1
    def view(self, path):
        log_ind = os.path.splitext(path)[0].split(sep='/')[-1].split(sep='_')[-1]
        new_window = tk.Toplevel(self.master)
        _figinds = [self._intcount, 100+self._intcount]
        self.viewers.append(VLA(new_window, path, 'FFBPlog_{}'.format(log_ind), _figinds))
        self._intcount += 1


def main():
    proceed = True
    VEX = ViewerExecutive(tk.Tk())
    while proceed:
        path = input('[FFBP Viewer] Enter name of log directory OR corresponding index: ')
        try:
            int(path)
            path = 'FFBP/logs/FFBPlog_{}/snap.pkl'.format(path)
        except ValueError:
            path = 'FFBP/logs/' + path
        with open(path) as snap:
            VEX.view(pickle.opensnap)
        print('[FFBP Viewer] Would you like to proceed?')
        prompt = input("[y/n] -> ")
        if prompt == 'n': proceed = False

if __name__=='__main__': main()
