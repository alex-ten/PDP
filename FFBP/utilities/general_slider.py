import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def slider_plot(data, func, xlab, ylab, note,):
    matplotlib.style.use('ggplot')

    fig, ax = plt.subplots()  # create a figure and axis (i.e. subplot) instances
    plt.subplots_adjust(left=0.25, bottom=0.25) # adjust subplot position
    x = data[:,0].astype(int) # range of 400 values between 0 and 4
    y_vec = data[:,1]
    y = func(y_vec, x) # define relationships between x and y
    line, = plt.plot(x, y, lw=2, color='#3DB1FF') # draw s as a function of t
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    # Set initial state of the slider graph
    hor = plt.axhline(y[-1], c='black', lw='0.8', ls=':') # add horizontal line
    ver = plt.axvline(x[-1], c='black', lw='0.8', ls=':') # add vertical line

    # place annotation near the last (x,y) point
    ann = ax.annotate('{}: {}'.format(note, np.around(y[-1],5)),  # annotation string
                      # annotation coordinates:
                      xy=(x[-1], y[-1]),  # position of element to annotate
                      xycoords='data',  # use xy to define the position in the coordinate system being annotated
                      xytext=(28, 26),  # position of the annotation
                      textcoords='offset points',  # specify an offset from the xy coordinates
                      # text properties
                      size=12,  # size in points
                      verticalalignment="center",  # position of text along the vertical axis of the box [aka va]
                      horizontalalignment="center",  # position of text along the horizontal axis of the box [aka ha]
                      bbox=dict(boxstyle="round",  # box shape (alt. square)
                                facecolor="#66B8FF",
                                edgecolor="none",
                                alpha=0.5),
                      arrowprops=dict(arrowstyle="wedge, tail_width=0.3, shrink_factor=1",
                                      # if arrowstyle key is present FancyArrowPatch prop dict is used
                                      fc="#66B8FF",
                                      ec="none",
                                      alpha=0.3,
                                      patchA=None,
                                      patchB=None,
                                      relpos=(0.5, 0.00),
                                      )
                      )

    # set max and min values for axes [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = [x.min(), x.max(), y.min(), y.max()]
    plt.axis([xmin,
              xmax + xmax * 0.05,
              ymin,
              ymax + ymax * 0.05])

    slide1_background_color = '#BDE9FF' # set the slide background color
    slide1_axis = plt.axes([0.25, 0.12, 0.65, 0.03], axisbg=slide1_background_color) # assign position for axes for slider

    # create a slider instance for AMPLITUDE using axamp from above
    slide1 = Slider(slide1_axis, xlab, x[0], x[-1], valinit=x[-1], fc='#588CBF', valfmt='%0.0f')

    def update(val):
        _x = slide1.val # assign value of the slider to _x (the dynamic value of x)
        _y = func(y_vec,_x) # dynamic value of _y is defined my the same relationship function
        ann.xy = (_x,_y) # set annotation xy to new values
        ann.set_text('{}: {}'.format(note, np.around(_y,5))) # change text accordingly
        # change positions of straight lines
        hor.set_ydata(_y)
        ver.set_xdata(_x)
        fig.canvas.draw_idle() # draw 2D line

    slide1.on_changed(update) # when the slider value is changed, call update function with the new slider position

    plt.show()