import tkinter as tk
from FFBP.visualization.VisLayersApp import VisLayersApp as VLA
from FFBP.visualization.NetworkData import NetworkData
from PDPATH import PDPATH
import pickle
import os

class ViewerExecutive():
    def __init__(self, master):
        self.master = master
        self.master.state('withdrawn')
        self.viewers = {}
    def view(self, snap, ID):
        if ID not in self.viewers.keys():
            new_window = tk.Toplevel(self.master)
            self.viewers[ID] = VLA(new_window, snap, 30, 96)
        else:
            self.viewers[ID].catch_up(snap)


def main():
    proceed = True
    VEX = ViewerExecutive(tk.Tk())
    usrdir = input('[FFBP Viewer] Provide user directory (if any), or press \'enter\' to use default directory: ')
    usrdir = usrdir.strip()
    while proceed:
        path = input('[FFBP Viewer] Enter name of log directory OR corresponding index: ')
        # Get path to snap file
        try:
            ID = int(path)
            path = os.getcwd()+'/FFBP{}/logs/FFBPlog_{}/snap.pkl'.format('/'+ usrdir if len(usrdir) else '', path)
        except ValueError:
            ID = int(path.split(sep='_')[-1])
            path = os.getcwd()+'/FFBP{}/logs/'.format('/'+ usrdir if len(usrdir) else '') + path + '/snap.pkl'
        with open(path, 'rb'):
            snap = NetworkData(path)
            VEX.view(snap, ID)

        print('[FFBP Viewer] Would you like to proceed?')
        prompt = input("[y/n] -> ")
        if prompt == 'n': proceed = False

if __name__=='__main__': main()
