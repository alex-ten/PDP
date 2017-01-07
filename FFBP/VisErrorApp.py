from FFBP.visualization import visError as ve
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

class VisErrorApp():
    def __init__(self, master, figure, data,
                 xlab = 'x', ylab = 'y', note = 'f(x)'):
        self.master = master
        self.figure = figure
        self.data = data
        self.note = note

        self.ax = figure.add_subplot(111)
        self.ax.set_xlabel(xlab)
        self.ax.set_ylabel(ylab)
        self.line, = self.ax.plot(data, lw = 2, color = '#3DB1FF')
        self.fann, self.hline, self.vline = \
            ve.fancy_annotation(self.ax,
            x = len(data) - 1,
            y = data[-1],
            note = note)

        self.canvasFrame = ttk.Frame(master, width = 500)
        self.Renderer = FigureCanvasTkAgg(figure, master)
        self.mplCanvas = self.Renderer.get_tk_widget()
        self.Renderer.draw()

        self.controlsFrame = ttk.Frame(master, width = 500)
        self.slide = ttk.Scale(self.controlsFrame,
                               orient = tk.HORIZONTAL,
                               length = 250,
                               value = len(data) - 1,
                               from_ = 0,
                               to = len(data) - 1,
                               command = self.onSlide)

        self.slide.set(str(len(data) - 1))

        self.continueButton = ttk.Button(self.controlsFrame,
                                         text = 'Continue', command = self.onContinue)

        self.canvasFrame.pack()
        self.mplCanvas.pack(fill = tk.BOTH, expand = True)
        self.controlsFrame.pack(fill = tk.X)
        self.slide.pack()
        self.continueButton.pack(fill = tk.X)
        self.master.mainloop()

    def plotLatest(self, data):
        self.data = data
        x, y = len(data) - 1, data[-1]
        self.line, = self.ax.plot(data, lw = 2, color = '#3DB1FF')
        self.fann.xy = (x, y)  # set annotation xy to new values
        self.fann.set_text('{}: {}'.format(self.note, np.around(y, 5)))  # change text accordingly
        # change positions of straight lines
        self.hline.set_ydata(y)
        self.vline.set_xdata(x)
        self.slide.config(to = float(x))
        self.slide.set(x)
        self.Renderer.draw()

    def onSlide(self, val):
        x = int(float(self.slide.get()))
        y = self.data[x]
        self.fann.xy = (x, y) # set annotation xy to new values
        self.fann.set_text('{}: {}'.format(self.note, np.around(y,5))) # change text accordingly
        # change positions of straight lines
        self.hline.set_ydata(y)
        self.vline.set_xdata(x)
        self.Renderer.draw()

    def onContinue(self):
        self._sleep()

    def catch_up(self, data):
        self.plotLatest(data)
        self.master.state('normal')
        self.master.mainloop()

    def _sleep(self):
        self.master.state('withdrawn')
        self.master.quit()


def main():
    from FFBP.visualization.NetworkData import NetworkData
    import matplotlib.pyplot as plt

    path = input('[visErrorApp.py] Path: ')
    snap = NetworkData(path)
    y = [x for x in snap.error]
    y1 = y[0:4]
    y2 = y

    fig = plt.figure(1)

    root = tk.Tk()

    VisApp = VisErrorApp(root, fig, y1, 'Epoch', 'Error', 'tss')
    input('[visErrorApp.py] hit enter to continue ')

    import time
    for i in range(3):
        print('[VisApp.py] Performing task {}'.format(i))
        time.sleep(1)
    VisApp.catch_up(y2)

if __name__=='__main__': main()