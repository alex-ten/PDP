import os
import pickle
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from collections import OrderedDict
from PDPATH import PDPATH

path_to_wordsr = PDPATH() + '/MIA/raw/wordsr.txt'

xmax = 4
ymax = 6

xoff = xmax/10
yoff = ymax/10

verts = ((0,3),(3,6),(6,8),(8,5),(5,2),(2,0), # outer O
         (3,4),(4,7),(4,5),(1,4), # inner +
         (6,4),(8,4),(2,4),(0,4)) # inner X

f_coords = (
    (0,0), (xmax/2,0), (xmax,0),
    (0,ymax/2), (xmax/2,ymax/2), (xmax,ymax/2),
    (0,ymax), (xmax/2,ymax), (xmax,ymax)
)


def vert2coord(v1, v2):
    x0, y0 = f_coords[v1]
    x1, y1 = f_coords[v2]
    return (x0+xoff, x1+xoff), (y0+yoff, y1+yoff)


def make_line(ax, xy, on=0):
    lw = 3
    if on:
        lc = 'k'
        z = 10
        a = on
    else:
        lc = 'k'
        z = 0
        a = 0.05
    cs = 'round'
    f0 = Line2D(xy[0], xy[1], linewidth=lw, color=lc, solid_capstyle=cs, zorder=z, alpha = a)
    ax.add_line(f0)


def remove_ticks(ax):
    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
    )


def draw_features(x, ax):
    ax.set_ylim(0, ymax + yoff * 2)
    ax.set_xlim(0, xmax + xoff * 2)
    ax.axis('off')
    for v, f0, f1 in zip(verts,np.array(x)[0::2], np.array(x)[1::2]):
        if f1: f = 1
        elif f0+f1 == 0: f = 0.3
        else: f = 0
        make_line(ax, vert2coord(v[0], v[1]), on=f)


wordlabs = []
with open(path_to_wordsr, 'r') as words_file:
    for line in words_file:
        wordlabs.append(line.replace(' ', '').upper().replace('\n', ''))


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
        self.viewers.append(MIAViewer(new_window, path, 'MIAlog_{}'.format(log_ind), _figinds))
        self._intcount += 1


class MIAViewer():
    def __init__(self, master, path, sim, inds):
        print('[MIA_Viewer] Initializing viewer for {} ...'.format(sim))
        self.master = master
        i1, i2 = inds
        # ============================= DATA =============================
        # ----------------------------- MAIN -----------------------------
        with open(path, 'rb') as logfile:
            self.data = pickle.load(logfile)

        self.master.title(sim)
        self.master.resizable(width = False, height=False)
        timesteps = len(self.data['word_mean'])
        self.figure = plt.figure(i1, facecolor='w', figsize=[10, 8])
        self.figure.clear()
        self.yax = np.fliplr([np.arange(26) + 0.5])[0]
        self.xax = np.fliplr([np.arange(36) + 0.5])[0]
        self.letlabs = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        plt.subplots_adjust(wspace=0.9, left=0.03, right=.90)
        self.letter_axes = OrderedDict()
        self.letter_axxes = OrderedDict()
        self.feat_disp = []

        self.minifig = plt.figure(i2, facecolor='w', figsize=[3, 0.8])
        self.minifig.clear()

        for i in range(3):
            vec = self.data['L{}_mean'.format(i)][0]
            vec_lab = ['{1} | {0}'.format(j, k) for j, k in zip(np.around(self.data['L{}_mean'.format(i)][0], 3),
                                                                np.around(self.data['L{}_marginal'.format(i)], 3))]
            ax = self.figure.add_subplot(1, 4, i + 1)
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
            axx.set_yticklabels(vec_lab)
            bars = ax.barh(self.yax, vec, align='center', alpha=1, facecolor='#FFB12B', linewidth=0)
            self.letter_axes[ax] = bars
            self.letter_axxes[axx] = None

        word_vec = self.data['word_mean'][0]
        word_vec_labs = ['[{1}] {0}'.format(j, k) for j, k in
                         zip(np.around(self.data['word_mean'][0], 3), np.around(self.data['word_marginal'], 3))]
        self.word_ax = self.figure.add_subplot(1, 4, 4)
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
        self.word_axx.set_yticklabels(word_vec_labs)
        self.word_bars = self.word_ax.barh(self.xax, word_vec, align='center', alpha=0.6, facecolor='#0D6EFF',
                                           linewidth=0)

        # ---------------------------- INPUT ----------------------------
        for i, s in enumerate(self.data['input']):
            self.mini_ax = self.minifig.add_subplot(1, 3, i + 1, aspect='equal')
            self.mini_ax.clear()
            draw_features(s, self.mini_ax)
            self.feat_disp.append(self.mini_ax)

        # =========================== WIDGETS ===========================
        # ---------------------------   MPL1   --------------------------
        self.canvasFrame = ttk.Frame(self.master, width=1200)
        self.tkCanvas = FigureCanvasTkAgg(self.figure, self.canvasFrame)
        self.tkCanvas.draw()
        self.mplCanvas = self.tkCanvas.get_tk_widget()

        # --------------------------- TKINTER ----------------------------
        # Controls:
        self.controlsFrame = ttk.Frame(self.master, width=1200, height=100)

        # Labels:
        self.time0Label = ttk.Label(self.controlsFrame, text='0')
        self.time1Label = ttk.Label(self.controlsFrame, text=str(int(timesteps - 1)))
        self.curTnSLabel = ttk.Label(self.controlsFrame, text='Time step: 0\nSum path prob: {}'.format(round(self.data['sumpp'],11)))
        self.bsLabel = ttk.Label(self.controlsFrame,
                                 text='Batch size: {}\nW to L scale factor: {}\nL to F scale factor: {}'.format(
                                     self.data['batch_size'], round(self.data['w2l'], 3), round(self.data['l2f'], 3)))

        # Slider and Button:
        self.slide = ttk.Scale(self.controlsFrame,
                               orient=tk.HORIZONTAL,
                               length=400,
                               value=0,
                               from_=0,
                               to=timesteps - 1,
                               command=self.onSlide)
        self.slide.set('0')
        self.closeButton = ttk.Button(self.controlsFrame, text='Close', command=self.onClose)

        # ---------------------------   MPL2   --------------------------
        self.miniCanvasFrame = ttk.Frame(self.controlsFrame, width=20)
        self.miniRenderer = FigureCanvasTkAgg(self.minifig, self.controlsFrame)
        self.miniMplCanvas = self.miniRenderer.get_tk_widget()
        self.miniRenderer.draw()

        # ========================== GEOMETRY ===========================
        self.canvasFrame.pack(fill=tk.BOTH)
        self.mplCanvas.pack(fill=tk.BOTH, expand=True)

        self.controlsFrame.pack(fill=tk.X, side=tk.BOTTOM)
        self.miniCanvasFrame.pack(side=tk.LEFT, pady=10)
        self.miniMplCanvas.pack(side=tk.LEFT, pady=10)

        self.time0Label.pack(side=tk.LEFT, pady=10, padx=10)
        self.slide.pack(side=tk.LEFT, pady=10)
        self.time1Label.pack(side=tk.LEFT, pady=10, padx=10)
        self.curTnSLabel.pack(side=tk.LEFT, pady=10, padx=30)
        self.bsLabel.pack(side=tk.LEFT, pady=10, padx=30)
        self.closeButton.pack(side=tk.RIGHT, pady=10, padx=10)
        self.master.protocol('WM_DELETE_WINDOW', self.onMasterX)

        for inp, ax in zip(self.data['input'], self.feat_disp):
            draw_features(inp, ax)
        self.miniRenderer.draw()

    def onSlide(self, val):
        x = int(float(self.slide.get()))

        letter_data = []
        for i, axx in enumerate(self.letter_axxes.keys()):
            let_vals = self.data['L{}_mean'.format(i)][x]
            vec_labs = ['[{1}] {0}'.format(j, k) for j, k in zip(np.around(self.data['L{}_mean'.format(i)][x],3),
                                                                np.around(self.data['L{}_marginal'.format(i)],3))]
            axx.set_yticklabels(vec_labs)
            letter_data.append(let_vals)

        word = self.data['word_mean'][x]
        word_vec_labs = ['[{1}] {0}'.format(j, k) for j, k in zip(np.around(self.data['word_mean'][x],3),
                                                                    np.around(self.data['word_marginal'],3))]
        self.word_axx.set_yticklabels(word_vec_labs)

        for i, bars in enumerate(self.letter_axes.values()):
            for vb, l in zip(bars, letter_data[i]):
                vb.set_width(l)

        for hb, w in zip(self.word_bars, word):
            hb.set_width(w)

        self.curTnSLabel.config(text='Time step: {}\nSum path prob: {}'.format(x, round(self.data['sumpp'],11)))
        self.tkCanvas.draw()

    def onClose(self):
        self._close()

    def onMasterX(self):
        self._close()

    def _close(self):
        self.master.quit()
        self.master.destroy()


# def main():
#     proceed = True
#     VEX = ViewerExecutive(tk.Tk())
#     while proceed:
#         path =  input('[MIA_Viewer] Enter name of log file OR log file index: ')
#         try:
#             int(path)
#             path = 'MIA/logs/MIAlog_{}.pkl'.format(path)
#         except ValueError:
#             path = 'MIA/logs/' + path
#         VEX.view(path)
#         print('[MIA_Viewer] Would you like to proceed? [return / n]')
#         prompt = input(">>> " )
#         if prompt == 'n': proceed = False
#
# if __name__=='__main__': main()