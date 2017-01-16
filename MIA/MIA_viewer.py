import tkinter as tk
from MIA.classes.MIAViewer import ViewerExecutive

def main():
    proceed = True
    VEX = ViewerExecutive(tk.Tk())
    while proceed:
        path = input('[MIA_Viewer] Enter name of log file OR log file index: ')
        try:
            int(path)
            path = 'MIA/logs/MIAlog_{}.pkl'.format(path)
        except ValueError:
            path = 'MIA/logs/' + path
        VEX.view(path)
        print('[MIA_Viewer] Would you like to proceed?')
        prompt = input("[y/n] -> ")
        if prompt == 'n': proceed = False

if __name__=='__main__': main()