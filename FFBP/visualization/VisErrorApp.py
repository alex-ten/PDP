from FFBP.visualization import visError as ve
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
                                y = round(data[-1],4),
                                note = note)

        self.canvasFrame = ttk.Frame(master, width = 500)
        self.Renderer = FigureCanvasTkAgg(figure, master)
        self.mplCanvas = self.Renderer.get_tk_widget()
        self.Renderer.draw()

        self.controlsFrame = ttk.Frame(master, width = 500, height = 40)
        self.slide = ttk.Scale(self.controlsFrame,
                               orient = tk.HORIZONTAL,
                               length = 250,
                               value = len(data) - 1,
                               from_ = 0,
                               to = len(data) - 1,
                               command = self.onSlide)
        self.zeroEpochLabel = ttk.Label(master = self.controlsFrame,
                                        text = '0')
        self.lastEpochLabel = ttk.Label(master = self.controlsFrame,
                                        text = str(len(data) - 1))
        self.currEpochLabel = ttk.Label(master = self.controlsFrame,
                                        text = 'epoch: {}'.format(int(self.slide.get())))
        self.slide.set(str(len(data) - 1))

        self.canvasFrame.pack()
        self.mplCanvas.pack(side = tk.TOP, fill = tk.X, expand = True)

        self.controlsFrame.pack(side = tk.TOP, expand = True)
        self.currEpochLabel.grid(row = 0, column = 1, columnspan = 3)
        self.zeroEpochLabel.grid(row = 1, column = 0, columnspan = 1, sticky = 'w')
        self.lastEpochLabel.grid(row = 1, column = 4, columnspan = 1, sticky = 'e')
        self.slide.grid(row = 1, column = 1, columnspan = 3)

        # ============================== PROTOCOLS ===============================
        self.master.protocol('WM_DELETE_WINDOW', self.onMasterX)

    def plotLatest(self, data):
        self.data = data
        x, y = len(data) - 1, data[-1]
        self.lastEpochLabel.config(text = str(x))
        self.line, = self.ax.plot(data, lw = 2, color = '#3DB1FF')
        self.fann.xy = (x, y)  # set annotation xy to new values
        self.fann.set_text('{}: {}'.format(self.note, y))  # change text accordingly
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
        self.fann.set_text('{}: {}'.format(self.note, y)) # change text accordingly

        # change positions of straight lines
        self.hline.set_ydata(y)
        self.vline.set_xdata(x)

        # Update epoch label
        self.currEpochLabel.config(text = 'epoch: {}'.format(x))
        self.Renderer.draw()

    def catch_up(self, data):
        self.master.state('normal')
        self.plotLatest(data)

    def onMasterX(self):
        self._sleep()

    def _sleep(self):
        self.master.withdraw()
        self.master.quit()

    def destroy(self):
        self.master.withdraw()
        self.master.quit()
        self.master.destroy()


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
