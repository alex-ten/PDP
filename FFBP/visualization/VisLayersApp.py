import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from FFBP.visualization import visLayers as vl
from FFBP.visualization.NetworkData import NetworkData


class VisLayersApp():
    def __init__(self, master, fig, snap, ppc, colors='coolwarm'):
        self.colors = colors
        self.master = master
        self.master.title('Network Visualization')
        # =========================== Figure preparation ===========================
        self.bfs = ppc * 0.4
        self.snap = snap
        self.panelBoard = vl.prep_figure(self.snap, fig)
        figSize = fig.get_size_inches() * fig.dpi

        # ================================= Style ==================================


        # ============================= Parent Windows =============================
        # ---------------------- Figure and window parameteres----------------------

        figWidth, figHeight = [int(x) for x in figSize]
        maxWindWidth = 900
        maxWindHeight = 700
        w = min(figWidth, maxWindWidth)
        h = min(figHeight, maxWindHeight)
        master.geometry(
            '{}x{}+0+0'.format(
                w+20,
                h))

        # --------------------------- Canvas parameteres ---------------------------
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)

        self.canvasFrame = ttk.Frame(master, width = w+20, height = h+20)
        self.canvasFrame.grid(row=0, column=0, sticky='nsew')
        self.canvasFrame.rowconfigure(0, weight=1)
        self.canvasFrame.columnconfigure(0, weight=1)

        self.backCanvas = tk.Canvas(self.canvasFrame, scrollregion=(0, 0, w, h))
        self.backCanvas.grid(row=0, column=0, sticky='nsew')

        self.yScroll = tk.Scrollbar(self.canvasFrame,
                                    orient=tk.VERTICAL,
                                    command=self.backCanvas.yview)
        self.xScroll = tk.Scrollbar(self.canvasFrame,
                                    orient=tk.HORIZONTAL,
                                    command=self.backCanvas.xview)
        self.yScroll.grid(row=0, column=1, sticky='ns')
        self.xScroll.grid(row=1, column=0, sticky='ew')
        self.backCanvas.config(xscrollcommand=self.xScroll.set,
                               yscrollcommand=self.yScroll.set)

        self.figureRenderer = FigureCanvasTkAgg(fig, self.backCanvas)
        self.mplCanvasWidget = self.figureRenderer.get_tk_widget()
        self.mplCanvasWidget.grid(sticky = 'nsew')
        # -------------------------------- Geometry --------------------------------

        self.backCanvas.create_window(0, 0, window=self.mplCanvasWidget)
        self.backCanvas.config(scrollregion=self.backCanvas.bbox(tk.ALL))
        self.figureRenderer.mpl_connect('pick_event', self.onPick)


        # ================================ CONTROLS ================================
        # --------------------------- Window parameteres ---------------------------
        controlsWidth = 230
        controlsHeight = 340
        tlx = max([0, w - controlsWidth - 30])
        tly = max([0, h - controlsHeight - 30])
        self.controlsWindow = tk.Toplevel(master)
        self.controlsWindow.title('Controls')
        self.controlsWindow.lift(master)
        self.controlsWindow.resizable('False','False')
        self.controlsWindow.geometry(
            '{}x{}+{}+{}'.format(
                controlsWidth,
                controlsHeight,
                tlx,
                tly
            )
        )
        # --------------------------------- Frames ---------------------------------

        # Info
        self.infoFrame = ttk.Frame(self.controlsWindow, width = 230, height = 115)
        self.epochSubFrame = ttk.Frame(self.infoFrame,
                                       width = 111,
                                       height = 105,
                                       relief = tk.GROOVE)
        self.cellSubFrame = ttk.Frame(self.infoFrame,
                                      width=111,
                                      height=105,
                                      relief = tk.GROOVE)
        self.tinySub = ttk.Frame(self.cellSubFrame,
                                 width = 40,
                                 height = 40,
                                 relief = tk.SUNKEN)
        # Slide and combobox
        self.selectorFrame = ttk.Frame(self.controlsWindow,
                                       width = 230,
                                       height = 95)
        # Buttons
        self.buttonFrame = ttk.Frame(self.controlsWindow)
        # Progress bar
        self.progressFrame = ttk.Frame(self.controlsWindow,
                                       width = 230,
                                       height = 20)

        # -------------------------------- tk Vars ---------------------------------
        self.patternVar = tk.StringVar()

        # -------------------------------- Widgets ---------------------------------

        # Selectors:
        self.patternSelector = ttk.Combobox(self.selectorFrame,
                                            textvariable = self.patternVar,
                                            values = list(snap.inp_names.keys()))
        self.patternSelector.current(0)

        self.epochSlider = ttk.Scale(self.selectorFrame,
                                     orient = tk.HORIZONTAL,
                                     length = 200,
                                     value =len(snap.epochs) - 1,
                                     from_ = 0,
                                     to =len(snap.epochs) - 1)

        self.epochSlider.set(str(len(snap.epochs) - 1))

        # Buttons
        self.updateButton = ttk.Button(self.buttonFrame,
                                       text='Update',
                                       command=self.onUpdate)

        self.labelsButton = ttk.Button(self.buttonFrame,
                                       text='Hide labels',
                                       command=self.onLabels)

        self.continueButton = ttk.Button(self.buttonFrame,
                                         text='Continue',
                                         command=self.onContinue)
        self.quitButton = ttk.Button(self.buttonFrame,
                                     text = 'Quit',
                                     command = self.onQuit)

        # Labels:
        #   epoch info
        self.epochValLabel = ttk.Label(self.epochSubFrame,
                                       text = str(self._get_epoch(int(self.epochSlider.get()))),
                                       font = ('Menlo', 30),
                                       justify = tk.CENTER)
        self.epochSlider.config(command = self.onSlide)

        self.epochLabel = ttk.Label(self.epochSubFrame,
                                    text = 'epoch',
                                    font = ('Menlo', 11),
                                    justify = tk.CENTER)

        #   draw cell onto tiny canvas
        self.tinyFig = plt.figure(3, figsize=(40 / 96, 40 / 96), facecolor='white')
        self.tinyRenderer = FigureCanvasTkAgg(self.tinyFig, self.tinySub)
        self.tinyCanvas = self.tinyRenderer.get_tk_widget()
        self.tinyRenderer.draw()

        self.cellWeight = ttk.Label(self.cellSubFrame,
                                    text = '-',
                                    font = ('Menlo', 11),
                                    justify = tk.CENTER)

        self.cellCoords = ttk.Label(self.cellSubFrame,
                                    text = 'r: - | c: -',
                                    font = ('Menlo', 11),
                                    justify = tk.CENTER)

        # Progress bar:
        self.progBar = ttk.Progressbar(self.progressFrame,
                                       orient = tk.HORIZONTAL,
                                       length = 200)

        # -------------------------------- Geometry --------------------------------

        # Info
        self.infoFrame.pack(side = tk.TOP, fill = tk.BOTH)

        # Epoch info
        self.epochSubFrame.pack(side = tk.LEFT, padx = 2, pady = 10)
        self.epochValLabel.place(relx = 0.50, rely = 0.45, anchor = tk.CENTER)
        self.epochLabel.place(relx = 0.50, rely = 0.75, anchor = tk.CENTER)

        # Cell info
        self.cellSubFrame.pack(side = tk.RIGHT, padx = 2, pady = 10)
        self.tinySub.place(relx = 0.50, rely = 0.40, anchor = tk.CENTER)
        self.tinyCanvas.pack()
        self.cellWeight.place(relx = 0.50, rely = 0.70, anchor = tk.CENTER)
        self.cellCoords.place(relx = 0.50, rely = 0.84, anchor = tk.CENTER)

        # Selectors
        self.selectorFrame.pack(fill = tk.BOTH)
        self.patternSelector.pack(fill = tk.Y, expand = True, pady=5)
        self.epochSlider.pack(fill = tk.Y, expand = True, pady=5)
        #
        # Buttons
        self.buttonFrame.pack(fill = tk.BOTH, expand = True, ipadx = 4)
        self.labelsButton.pack(fill = tk.X, padx = 10)
        self.updateButton.pack(fill = tk.X, padx = 10)
        self.continueButton.pack(fill = tk.X, padx = 10)
        self.quitButton.pack(fill = tk.X, padx = 10)

        # Progress Bar
        self.progressFrame.pack(fill = tk.BOTH, expand = True)
        self.progBar.place(relx = 0.5, rely = 0.5, anchor = tk.CENTER)
        self.progBar.config(mode='indeterminate')

        # ============================ Initial Figure ============================
        self._label_groups = vl.annotate(self.snap, self.panelBoard, self.bfs)
        self._labels_on = True
        self._plotLatest()

        self.master.mainloop()

    def onUpdate(self):
        epoch_ind = int(self.epochSlider.get())
        key = self.patternSelector.get()
        if key in self.snap.inp_names.keys():
            self.progBar.start()
            self.panelBoard.clear()
            ind_map = self.snap.inp_vects[epoch_ind]
            pattern_ind = np.where(np.all(ind_map == self.snap.inp_names[key], axis=1))[0][0]
            vl.draw_all_layers(self.snap, self.panelBoard, epoch_ind, pattern_ind, colmap=self.colors)
            self._label_groups = vl.annotate(self.snap, self.panelBoard, self.bfs)
            self.figureRenderer.draw()
            self.progBar.stop()
        else:
            messagebox.showinfo(title='Wrong selection',
                                message='No such pattern. Please select a pattern from the list')

    def onLabels(self):
        if self._labels_on:
            vl.labels_off(self._label_groups)
            self._labels_on = False
            self.labelsButton.config(text = 'Show labels')
            self.figureRenderer.draw()
        else:
            self._label_groups = vl.annotate(self.snap, self.panelBoard, self.bfs)
            self._labels_on = True
            self.labelsButton.config(text='Hide labels')
            self.figureRenderer.draw()

    def onContinue(self):
        self._sleep()

    def onQuit(self):
        self.controlsWindow.destroy()
        self.master.destroy()
        self.master.quit()


    def onPick(self, event):
        thiscell = event.artist
        value = thiscell.get_cellval()
        r, c = thiscell.get_inds()
        weight = str(round(value, 4))
        rc = 'r: {} | c: {}'.format(r,c)
        self.cellCoords.config(text = rc)
        self.cellWeight.config(text = weight)
        self.tinyFig.set_facecolor(vl.v2c(value, self.colors))
        self.tinyRenderer.draw()

    def onSlide(self, val):
        val = float(val)
        self.epochValLabel.config(text = str(self._get_epoch(val)))

    def catch_up(self, snap):
        self.snap = snap
        self._plotLatest()
        self.master.state('normal')
        self.controlsWindow.state('normal')
        self.master.mainloop()

    def _sleep(self):
        self.master.withdraw()
        self.controlsWindow.withdraw()
        self.master.quit()

    def _plotLatest(self):
        latest_epoch_ind = len(self.snap.epochs) - 1
        vl.draw_all_layers(self.snap,
                           self.panelBoard,
                           latest_epoch_ind,
                           0)
        self.epochSlider.config(to = float(latest_epoch_ind))
        self.figureRenderer.draw()

    def _get_epoch(self, slider_value):
        return self.snap.epochs[int(slider_value)]

    def _get_pattern(self):
        try:
            ind = self.snap.inp_names.index(self.patternVar.get())
            print('You selected pattern {}'.format(ind))
        except ValueError:
            print('No such pattern')




def main(ppc = 20, screen_dpi = 96):
    path = input('[VisApp.py] Snapshot path: ')

    # Get network data
    snap = NetworkData(path)

    # Create a figure
    max_width = max([l.sender[1] for l in snap.main.values()])
    width_cells = ((max_width + 9) * 2)
    width_pixels =  width_cells * ppc
    width_inches = width_pixels  / screen_dpi

    network_size = snap.num_units
    height_cells = network_size + (6 * snap.num_layers)
    height_pixels = height_cells * ppc
    height_inches = height_pixels / screen_dpi

    print('''
Figure dimensions (ppc = {} | dpi = {}):
-------------------------------
 Units:    width x height
-------------------------------
 Cells:    {} x {}
 Pixels:   {} x {}
 Inches:   {} x {}
-------------------------------
    '''.format(
        ppc, screen_dpi,
        width_cells, height_cells,
        width_pixels, height_pixels,
        round(width_inches, 3), round(height_inches, 3)
        )
    )

    fig = plt.figure(2, figsize=(width_inches, height_inches), facecolor='w', dpi=screen_dpi)

    # Start the app
    root = tk.Tk()
    app = VisLayersApp(root, fig, snap, 30)
    input('[VisApp.py] Hit enter to start a parallel process')

    import time
    for i in range(5):
        print('[VisApp.py] Performing task {}'.format(i))
        time.sleep(1)
    app.catch_up(path)

if __name__=='__main__': main()