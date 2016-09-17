import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from FFBP.visualization import visLayers as vl
from FFBP.visualization.NetworkData import NetworkData


class VisLayersApp():
    def __init__(self, master, snap, ppc, dpi, colors='coolwarm'):

        self.master = master
        self.master.title('Network Visualization')
        # =========================== Figure preparation ===========================
        self.colors = colors
        self._ppc = ppc
        self._dpi = dpi
        self.bfs = self._set_bfs()
        self.snap = snap
        self.figure = self.create_fig()
        self.panelBoard = vl.prep_figure(self.snap, self.figure)
        figSize = self.figure.get_size_inches() * self._dpi

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

        # --------------------------------- Canvas ---------------------------------
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)

        # Frames
        self.canvasFrame = ttk.Frame(master, width = w+20, height = h+20)
        self.canvasFrame.grid(row=0, column=0, columnspan = 2, sticky='nsew')
        self.canvasFrame.rowconfigure(0, weight=1)
        self.canvasFrame.columnconfigure(0, weight=1)

        # Widgets
        self.backCanvas = tk.Canvas(self.canvasFrame, scrollregion=(0, 0, w, h))
        self.backCanvas.grid(row=0, column=0, columnspan = 2, sticky='nsew')

        self.yScroll = tk.Scrollbar(self.canvasFrame,
                                    orient=tk.VERTICAL,
                                    command=self.backCanvas.yview)
        self.xScroll = tk.Scrollbar(self.canvasFrame,
                                    orient=tk.HORIZONTAL,
                                    command=self.backCanvas.xview)
        self.yScroll.grid(row=0, column=1, columnspan = 1, sticky='ns')
        self.xScroll.grid(row=1, column=0, columnspan = 1, sticky='ew')
        self.backCanvas.config(xscrollcommand=self.xScroll.set,
                               yscrollcommand=self.yScroll.set)

        self.figureRenderer = FigureCanvasTkAgg(self.figure, self.backCanvas)
        self.mplCanvasWidget = self.figureRenderer.get_tk_widget()
        self.mplCanvasWidget.grid(sticky = 'nsew')

        self.liftControlsButton = ttk.Button(self.master,
                                             text = 'Lift controls',
                                             command=self.onLiftControls)
        self.liftControlsButton.grid(row=2, column=0, sticky='ew')

        self.colprefsButton = ttk.Button(self.master,
                                         text = 'Color preferences',
                                         command = self.onColorPrefs)
        self.colprefsButton.grid(row=2, column=1, sticky='ew')

        # Integrate mpl and tk
        self.backCanvasWind = self.backCanvas.create_window(0, 0, window=self.mplCanvasWidget, anchor = 'nw')
        self.backCanvas.config(scrollregion=self.backCanvas.bbox(tk.ALL))
        self.figureRenderer.mpl_connect('pick_event', self.onPick)


        # ================================ CONTROLS ================================
        # --------------------------- Window parameteres ---------------------------
        controlsWidth = 230
        controlsHeight = 340
        self.controlsWindow = tk.Toplevel(master)
        self.controlsWindow.title('Controls')
        self.controlsWindow.lift(master)
        self.controlsWindow.resizable('False','False')
        self.controlsWindow.geometry(
            '{}x{}+0+0'.format(
                controlsWidth,
                controlsHeight
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
        self.continueFrame = ttk.Frame(self.controlsWindow)

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

        self.zoominButton = ttk.Button(self.buttonFrame,
                                       text = 'Zoom in',
                                       command = lambda: self.changeSize(1))

        self.zoomoutButton = ttk.Button(self.buttonFrame,
                                        text='Zoom out',
                                        command=lambda: self.changeSize(-1))

        self.continueButton = ttk.Button(self.continueFrame,
                                         text='Continue',
                                         command=self.onContinue)
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
        self.buttonFrame.pack(fill = tk.BOTH, expand = True)
        self.buttonFrame.columnconfigure(0,weight=1)
        self.buttonFrame.columnconfigure(1, weight=1)
        self.labelsButton.grid(row=0, column = 0, columnspan=2, sticky='ew', padx=10)
        self.zoominButton.grid(row=1, column = 1, columnspan=1, sticky='ew', padx=10)
        self.zoomoutButton.grid(row=1, column = 0, columnspan=1, sticky='ew', padx=10)
        self.updateButton.grid(row=2, column=0, columnspan=2, sticky='ew', padx=10)

        self.continueFrame.pack(fill=tk.X)
        self.continueButton.pack(fill = tk.X, padx = 10)

        # Progress Bar
        self.progressFrame.pack(fill = tk.BOTH, expand = True)
        self.progBar.place(relx = 0.5, rely = 0.5, anchor = tk.CENTER)
        self.progBar.config(mode='indeterminate')

        # ================================ COLORS ==================================
        mpl_color_maps = (['BrBG', 'bwr', 'coolwarm', 'PiYG',
                           'PRGn', 'PuOr','RdBu', 'RdGy',
                           'RdYlBu', 'RdYlGn', 'Spectral',
                           'seismic', 'jet', 'rainbow', 'terrain'])

        self.colorsWindow = tk.Toplevel(self.master, )
        self.colorsWindow.title('Color Preferences')
        self.colorsWindow.resizable('False','False')
        self.colorsWindow.geometry('{}x{}+0+0'.format(350, 150))
        self.colorsWindow.withdraw()

        self.colorsFrame = ttk.Frame(self.colorsWindow)
        self.colorsFrame.place(relx=0.5, rely=0.5, anchor = tk.CENTER, width = 350, height = 150)
        self.colorsFrame.columnconfigure(0, weight = 1)
        self.colorsFrame.columnconfigure(1, weight = 1)
        self.colorsFrame.columnconfigure(2, weight=1)

        self.normMode = tk.StringVar()
        self.normMode.set('cus')

        self.colMap = tk.StringVar()

        self.absRadio = ttk.Radiobutton(self.colorsFrame, variable = self.normMode, value = 'abs', text = 'Absolute')
        self.relRadio = ttk.Radiobutton(self.colorsFrame, variable = self.normMode, value = 'rel', text = 'Relative')
        self.cusRadio = ttk.Radiobutton(self.colorsFrame, variable = self.normMode, value = 'cus', text = 'Custom')

        self.colmapLabel = ttk.Label(self.colorsFrame, text='color map', font=('Helvetica', 10))
        self.nrangeLabel = ttk.Label(self.colorsFrame, text='nrange', font=('Helvetica', 10))

        self.colmapCombo = ttk.Combobox(self.colorsFrame, textvariable = self.colMap, values = mpl_color_maps)
        self.colmapCombo.set(self.colors)
        self.nrangeEntry = tk.Entry(self.colorsFrame, width = 7)
        self.nrangeEntry.insert(0, '1')

        self.applyButton = ttk.Button(self.colorsFrame, text = 'Apply', command = self.onApply)
        self.helpButton = ttk.Button(self.colorsFrame, text = '?', command = self.onHelp)


        self.absRadio.grid(row = 0, column = 0, columnspan = 1, padx = 15, pady = 10, sticky = 'w')
        self.relRadio.grid(row = 0, column = 1, columnspan = 1, padx = 15, pady = 10, sticky = 'w')
        self.cusRadio.grid(row = 0, column = 2, columnspan = 1, padx = 15, pady = 10, sticky = 'w')

        self.colmapLabel.grid(row = 1, column = 0, columnspan = 2, padx = 17, sticky = 'ws')
        self.nrangeLabel.grid(row = 1, column = 2, columnspan = 1, padx = 15, sticky = 'ws')

        self.colmapCombo.grid(row = 2, column = 0, columnspan = 2, padx = 15, sticky = 'ews')
        self.nrangeEntry.grid(row = 2, column = 2, columnspan = 1)

        self.applyButton.grid(row = 3, column = 1, columnspan = 2, padx = 15, pady = 20, sticky = 'nesw')
        self.helpButton.grid(row = 3, column = 0, columnspan = 1, padx = 15, pady = 20, sticky = 'w')

        # ============================ Initial Figure ============================
        self._label_groups = vl.annotate(self.snap, self.panelBoard, self._set_bfs())
        self._labels_on = True
        self._plotLatest()

        # ============================== PROTOCOLS ===============================
        self.master.protocol('WM_DELETE_WINDOW', self.onMasterX)
        self.controlsWindow.protocol('WM_DELETE_WINDOW', self.onControlsX)
        self.colorsWindow.protocol('WM_DELETE_WINDOW', self.onColorsX)

        self.master.mainloop()

    def create_fig(self):
        # Create a figure
        max_width = max([l.sender[1] for l in self.snap.main.values()])
        width_cells = ((max_width + 9) * 2)
        width_pixels = width_cells * self._ppc
        width_inches = width_pixels / self._dpi

        network_size = self.snap.num_units
        height_cells = network_size + (6 * self.snap.num_layers)
        height_pixels = height_cells * self._ppc
        height_inches = height_pixels / self._dpi
        fig = plt.figure(2, figsize=(width_inches, height_inches), facecolor='w', dpi=self._dpi)
        return fig

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
            self._label_groups = vl.annotate(self.snap, self.panelBoard, self._set_bfs())
            self._labels_on = True
            self.labelsButton.config(text='Hide labels')
            self.figureRenderer.draw()

    def onContinue(self):
        self._sleep()

    def changeSize(self, direction):
        oldSize_inches = self.figure.get_size_inches()
        oldSize_pixels = [s_i * self._dpi for s_i in oldSize_inches]
        Size_cells = [s_p / self._ppc for s_p in oldSize_pixels]
        self._ppc += 10 * direction
        newSize_pixels = [s_c * self._ppc for s_c in Size_cells]
        newSize_inches = [s_p / self._dpi for s_p in newSize_pixels]
        self.figure.set_size_inches(newSize_inches)
        nW, nH = newSize_pixels[0], newSize_pixels[1]
        self.canvasFrame.config(width = nW, height = nH)
        self.mplCanvasWidget.config(width = nW, height = nH)
        self.backCanvas.itemconfigure(self.backCanvasWind, width=nW, height=nH)
        self.backCanvas.config(scrollregion=self.backCanvas.bbox(tk.ALL), width=nW, height=nH)
        self.checkPPC()
        if self._labels_on:
            vl.labels_off(self._label_groups)
            self._label_groups = vl.annotate(self.snap, self.panelBoard, self._set_bfs())
            self.figureRenderer.draw()
        self.figure.canvas.draw()

    def checkPPC(self):
        upperlim = 60
        lowerlim = 10
        if self._ppc <= lowerlim:
            self.zoomoutButton.state(['disabled'])
        elif self._ppc >= upperlim:
            self.zoominButton.state(['disabled'])
        else:
            if self.zoomoutButton.instate(['disabled']): self.zoomoutButton.state(['!disabled'])
            if self.zoominButton.instate(['disabled']): self.zoominButton.state(['!disabled'])

    def onPick(self, event):
        thiscell = event.artist
        value = thiscell.get_cellval()
        r, c = thiscell.get_inds()
        weight = str(round(value, 4))
        rc = 'r: {} | c: {}'.format(r,c)
        self.cellCoords.config(text = rc)
        self.cellWeight.config(text = weight)
        self.tinyFig.set_facecolor(vl.v2c(value, self.colors, 1))
        self.tinyRenderer.draw()

    def onSlide(self, val):
        val = float(val)
        self.epochValLabel.config(text = str(self._get_epoch(val)))

    def onApply(self):
        print('Applying changes')
        print('Normalization scope: {}'.format(self.normMode.get()))
        print('Normalization range: {}'.format(self.nrangeEntry.get()))
        print('Color map: {}'.format(self.colmapCombo.get()))
        self.colorsWindow.withdraw()

    def onHelp(self):
        messagebox.showinfo('Need some explanation?',
                            'Too bad, we are still working on it. '
                            'Try playing around with the preferences '
                            'to see what\'s going on :)')

    def onLiftControls(self):
        if self.controlsWindow.state() == 'withdrawn': self.controlsWindow.state('normal')
        self.controlsWindow.lift()

    def onColorPrefs(self):
        if self.colorsWindow.state() == 'withdrawn': self.colorsWindow.state('normal')
        self.colorsWindow.lift()

    def onMasterX(self):
        self._sleep()

    def onControlsX(self):
        self.controlsWindow.withdraw()

    def onColorsX(self):
        positive = messagebox.askyesno('O_o', 'Do you want to apply changes')
        if positive:
            self.onApply()
        self.colorsWindow.withdraw()

    def catch_up(self, snap):
        self.snap = snap
        self._plotLatest()
        self.master.state('normal')
        self.controlsWindow.state('normal')
        self.master.mainloop()

    def _sleep(self):
        self.master.withdraw()
        self.controlsWindow.withdraw()
        self.colorsWindow.withdraw()
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

    def _set_bfs(self, factor=0.4):
        return self._ppc * factor




def main():

    #path = input('[VisApp.py] Snapshot path: ')

    # Get network data
    snap = NetworkData('/Users/alexten/Projects/PDP/FFBP/logdir/Sess_2016-09-17_16-44-47/mpl_data/snapshot_log.pkl')

    # Start the app
    root = tk.Tk()
    app = VisLayersApp(root, snap, 30, 96)
    input('[VisApp.py] Hit enter to start a parallel process')

    import time
    for i in range(2):
        print('[VisApp.py] Performing task {}'.format(i))
        time.sleep(1)
    app.catch_up(snap)

if __name__=='__main__': main()