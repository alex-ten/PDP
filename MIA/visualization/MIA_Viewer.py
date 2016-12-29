import os
import pickle
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from collections import OrderedDict

path_to_wordsr = '/Users/alexten/Projects/PDP/MIA/raw/wordsr.txt'

wordlabs = []
with open(path_to_wordsr, 'r') as words_file:
    for line in words_file:
        wordlabs.append(line.replace(' ', '').upper().replace('\n', ''))


class MIA_Viewer():
    def __init__(self, master, data):
        self.master = master
        self.figure = plt.figure(1, facecolor='w')
        self.data = data
        self.xax = np.fliplr([np.arange(26) + 0.5])[0]
        self.yax = np.fliplr([np.arange(36) + 0.5])[0]
        self.letlabs = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        plt.subplots_adjust(wspace=0.5, left=0.03, right=.95)
        self.letter_axes = OrderedDict()
        self.letter_data = []

        for i in range(3):
            vec = self.data['L{}_mean'.format(i)][0]
            ax = self.figure.add_subplot(1, 4, i+1)
            ax.set_title('Letter in position {}'.format(i))
            ax.set_ylim(0, 26)
            ax.set_xlim(0, 1)
            ax.set_yticks(self.xax)
            ax.set_yticklabels(self.letlabs)
            bars = ax.barh(self.xax, vec, align='center')
            self.letter_axes[ax] = bars

        self.word_ax = self.figure.add_subplot(1,4,4)
        self.word_ax.set_title('Word')
        self.word_ax.set_xlim(0, 1)
        self.word_ax.set_ylim(0, 36)
        self.word_ax.set_yticks(self.yax)
        self.word_ax.set_yticklabels(wordlabs)
        self.word_bars = self.word_ax.barh(self.yax, data['word_mean'][0], align='center')

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

        self.continueButton = ttk.Button(self.controlsFrame, text = 'Continue', command = self.onContinue)
        self.canvasFrame.pack()
        self.mplCanvas.pack(fill = tk.BOTH, expand = True)
        self.controlsFrame.pack(fill = tk.X)
        self.slide.pack()
        self.continueButton.pack(fill = tk.X)
        self.master.protocol('WM_DELETE_WINDOW', self.onMasterX)
        self.master.state('normal')
        self.master.mainloop()

    def plot_new_data(self, new_data):
        self.data = new_data
        self.slide.set('0')
        for i, ax in enumerate(self.letter_axes.keys()):
            ax.clear()
            vec = self.data['L{}_mean'.format(i)][0]
            ax.set_title('Letter in position {}'.format(i))
            ax.set_ylim(0, 26)
            ax.set_xlim(0, 1)
            ax.set_yticks(self.xax)
            ax.set_yticklabels(self.letlabs)
            self.letter_axes[ax] = ax.barh(self.xax, vec, align='center')
        self.word_ax.clear()

        self.word_ax.set_title('Word')
        self.word_ax.set_xlim(0, 1)
        self.word_ax.set_ylim(0, 36)
        self.word_ax.set_yticks(self.yax)
        self.word_ax.set_yticklabels(wordlabs)
        self.word_bars = self.word_ax.barh(self.yax, self.data['word_mean'][0], align='center')

        self.Renderer.draw()
        self.master.state('normal')
        self.master.mainloop()


    def onSlide(self, val):
        x = int(float(self.slide.get()))
        self.letter_data = []
        for i in range(3):
            self.letter_data.append(self.data['L{}_mean'.format(i)][x])
        word = self.data['word_mean'][x]

        for i, bars in enumerate(self.letter_axes.values()):
            for vb, l in zip(bars, self.letter_data[i]):
                vb.set_width(l)

        for hb, w in zip(self.word_bars, word):
            hb.set_width(w)

        self.Renderer.draw()

    def onContinue(self):
        self._sleep()

    def onMasterX(self):
        self._sleep()

    def _sleep(self):
        self.master.state('withdrawn')
        self.master.quit()


def main():
    first = True
    proceed = True
    while proceed:
        path = input('[MIA_Viewer] Provide path to log dict: ')
        root = tk.Tk()
        if first:
            first = False
            with open(path, 'rb') as f:
                data = pickle.load(f)
            VisApp = MIA_Viewer(root, data)
        else:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            VisApp.new_data(data)
        print('[MIA_Viewer] Would you like to proceed?')
        prompt = input('[y/n] -> ')
        if prompt == 'n': proceed = False


if __name__=='__main__': main()