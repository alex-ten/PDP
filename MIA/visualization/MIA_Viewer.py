import os
import pickle
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from collections import OrderedDict

path_to_wordsr = '/Users/alexten/Projects/PDP/MIA/raw/wordsr.txt'

def remove_ticks(ax):
    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
    )

wordlabs = []
with open(path_to_wordsr, 'r') as words_file:
    for line in words_file:
        wordlabs.append(line.replace(' ', '').upper().replace('\n', ''))


class MIA_Viewer():
    def __init__(self, master, data):
        self.master = master
        self.figure = plt.figure(1, facecolor='w', figsize=[10,8])
        self.data = data
        self.yax = np.fliplr([np.arange(26) + 0.5])[0]
        self.xax = np.fliplr([np.arange(36) + 0.5])[0]
        self.letlabs = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        plt.subplots_adjust(wspace=0.5, left=0.03, right=.95)
        self.letter_axes = OrderedDict()
        self.letter_axxes = OrderedDict()
        self.letter_data = []

        for i in range(3):
            vec = self.data['L{}_mean'.format(i)][0]
            ax = self.figure.add_subplot(1, 4, i+1)
            ax.set_title('Letter in position {}'.format(i))
            ax.set_ylim(0, 26)
            ax.set_xlim(0, 1)
            ax.set_yticks(self.yax)
            ax.set_yticklabels(self.letlabs)
            ax.yaxis.grid(True, linestyle='-', color='grey', alpha=0.5)
            remove_ticks(ax)
            axx = ax.twinx()
            axx.set_ylim(0, 26)
            axx.set_yticks(ax.get_yticks())
            axx.set_yticklabels(np.around(vec, 4))
            bars = ax.barh(self.yax, vec, align='center', alpha = 1, facecolor='#FFB12B', linewidth=0)
            self.letter_axes[ax] = bars
            self.letter_axxes[axx] = None

        word_vec = data['word_mean'][0]
        self.word_ax = self.figure.add_subplot(1,4,4)
        self.word_ax.set_title('Word')
        self.word_ax.set_xlim(0, 1)
        self.word_ax.set_ylim(0, 36)
        self.word_ax.set_yticks(self.xax)
        self.word_ax.set_yticklabels(wordlabs)
        self.word_ax.yaxis.grid(True, linestyle='-', color='grey', alpha=0.5, zorder=1)
        remove_ticks(self.word_ax)
        self.word_axx = self.word_ax.twinx()
        self.word_axx.set_ylim(0, 36)
        self.word_axx.set_yticks(self.word_ax.get_yticks())
        self.word_axx.set_yticklabels(np.around(word_vec, 4))
        self.word_bars = self.word_ax.barh(self.xax, word_vec, align='center', alpha=0.6, facecolor='#0D6EFF', linewidth=0)

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
        self.canvasFrame.pack(fill = tk.BOTH)
        self.mplCanvas.pack(fill = tk.BOTH, expand = True)
        self.controlsFrame.pack(fill = tk.X)
        self.slide.pack()
        self.continueButton.pack()
        self.master.protocol('WM_DELETE_WINDOW', self.onMasterX)
        self.master.state('normal')
        self.master.mainloop()

    def plot_new_data(self, new_data):
        self.data = new_data
        self.slide.set('0')

        for i, axs in enumerate(zip(self.letter_axes.keys(), self.letter_axxes.keys())):
            axs = list(axs)
            axs[0].clear()
            vec = self.data['L{}_mean'.format(i)][0]
            axs[0].set_title('Letter in position {}'.format(i))
            axs[0].set_ylim(0, 26)
            axs[0].set_xlim(0, 1)
            axs[0].set_yticks(self.yax)
            axs[0].set_yticklabels(self.letlabs)
            axs[0].yaxis.grid(True, linestyle='-', color='grey', alpha=0.5)
            remove_ticks(axs[0])
            axs[1].set_ylim(0, 26)
            axs[1].set_yticks(axs[0].get_yticks())
            axs[1].set_yticklabels(np.around(vec, 4))
            self.letter_axes[axs[0]] = axs[0].barh(self.yax, vec, align='center', facecolor='#FFB12B', linewidth=0)

        self.word_ax.clear()
        self.word_axx.clear()
        word_vec = self.data['word_mean'][0]
        self.word_ax.set_title('Word')
        self.word_ax.set_xlim(0, 1)
        self.word_ax.set_ylim(0, 36)
        self.word_ax.set_yticks(self.xax)
        self.word_ax.set_yticklabels(wordlabs)
        self.word_ax.yaxis.grid(True, linestyle='-', color='grey', alpha=0.5, zorder=1)
        remove_ticks(self.word_ax)
        self.word_axx.set_ylim(0, 36)
        self.word_axx.set_yticks(self.word_ax.get_yticks())
        self.word_axx.set_yticklabels(np.around(word_vec, 4))
        self.word_bars = self.word_ax.barh(self.xax, word_vec, align='center', alpha = 0.6, facecolor='#0D6EFF', linewidth=0)

        self.Renderer.draw()
        self.master.state('normal')
        self.master.mainloop()


    def onSlide(self, val):
        x = int(float(self.slide.get()))
        self.letter_data = []
        for i, axx in enumerate(self.letter_axxes.keys()):
            let_vals = self.data['L{}_mean'.format(i)][x]
            axx.set_yticklabels(np.around(let_vals, 3))
            self.letter_data.append(let_vals)
        word = self.data['word_mean'][x]
        self.word_axx.set_yticklabels(word)

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
        path = '/Users/alexten/Projects/PDP/logdir/Sess_2016-12-31_01-21-21/mpl_data/log_dict-GOD.pkl'
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