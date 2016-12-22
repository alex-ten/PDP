import os
import pickle
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
path_to_wordsr = os.getcwd()+'/MIA/raw/wordsr.txt'

wordlabs = []
with open(path_to_wordsr, 'r') as words_file:
    for line in words_file:
        wordlabs.append(line.replace(' ', '').upper().replace('\n', ''))


class MIA_Viewer():
    def __init__(self, master, data):
        self.master = master
        self.figure = plt.figure(1, facecolor='w')
        self.data = data
        xax = np.arange(26) + 0.5
        yax = np.fliplr([np.arange(36)+0.5])[0]
        letlabs = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        plt.subplots_adjust(hspace=0.5)

        self.L0_ax = self.figure.add_subplot(321)
        self.L0_ax.set_title('Letter in position 0')
        self.L0_ax.set_xlim(0, 26)
        self.L0_ax.set_ylim(0, 1)
        self.L0_ax.set_ylabel('')
        self.L0_ax.set_xticks(xax)
        self.L0_ax.set_xticklabels(letlabs)
        self.L0_bars = self.L0_ax.bar(xax, data['L0_mean'][0], align='center')

        self.L1_ax = self.figure.add_subplot(323)
        self.L1_ax.set_title('Letter in position 1')
        self.L1_ax.set_xlim(0, 26)
        self.L1_ax.set_ylim(0, 1)
        self.L1_ax.set_ylabel('L1')
        self.L1_ax.set_xticks(xax)
        self.L1_ax.set_xticklabels(letlabs)
        self.L1_bars = self.L1_ax.bar(xax, data['L1_mean'][0], align='center')

        self.L2_ax = self.figure.add_subplot(325)
        self.L2_ax.set_title('Letter in position 2')
        self.L2_ax.set_xlim(0, 26)
        self.L2_ax.set_ylim(0, 1)
        self.L2_ax.set_ylabel('L2')
        self.L2_ax.set_xticks(xax)
        self.L2_ax.set_xticklabels(letlabs)
        self.L2_bars = self.L2_ax.bar(xax, data['L2_mean'][0], align='center')

        self.word_ax = self.figure.add_subplot(122)
        self.word_ax.set_title('Word')
        # self.word_ax.set_xlim(0, 0.3)
        self.word_ax.set_ylim(0, 36)
        self.word_ax.set_yticks(yax)
        self.word_ax.set_yticklabels(wordlabs)
        self.word_bars = self.word_ax.barh(yax, data['word_mean'][0], align='center')

        self.canvasFrame = ttk.Frame(master, width = 1200)
        self.Renderer = FigureCanvasTkAgg(self.figure, master)
        self.mplCanvas = self.Renderer.get_tk_widget()
        self.Renderer.draw()

        self.controlsFrame = ttk.Frame(master, width = 1200)
        timesteps = len(data['word_mean'])
        self.slide = ttk.Scale(self.controlsFrame,
                               orient = tk.HORIZONTAL,
                               length = 400,
                               value = 0,
                               from_ = 0,
                               to = timesteps - 1,
                               command = self.onSlide)

        self.slide.set('0')

        self.continueButton = ttk.Button(self.controlsFrame,
                                         text = 'Continue', command = self.onContinue)

        self.canvasFrame.pack()
        self.mplCanvas.pack(fill = tk.BOTH, expand = True)
        self.controlsFrame.pack(fill = tk.X)
        self.slide.pack()
        self.continueButton.pack(fill = tk.X)
        self.master.mainloop()

    def onSlide(self, val):
        x = int(float(self.slide.get()))
        L0 = self.data['L0_mean'][x]
        L1 = self.data['L1_mean'][x]
        L2 = self.data['L2_mean'][x]
        word = self.data['word_mean'][x]

        for vb0, vb1, vb2, l0, l1, l2 in zip(self.L0_bars, self.L1_bars, self.L2_bars, L0, L1, L2):
            vb0.set_height(l0)
            vb1.set_height(l1)
            vb2.set_height(l2)

        for hb, w in zip(self.word_bars, word):
            hb.set_width(w)
        self.Renderer.draw()

    def onContinue(self):
        self._sleep()

    def _sleep(self):
        self.master.state('withdrawn')
        self.master.quit()


def main():
    # path = input('[visErrorApp.py] Path: ')
    path = '/Users/alexten/Projects/PDP/MIA/logdir/Sess_2016-12-20_20-05-08/mpl_data/log_dict.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    root = tk.Tk()
    VisApp = MIA_Viewer(root, data)

if __name__=='__main__': main()