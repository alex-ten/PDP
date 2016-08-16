import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def slider_plot(data, func, xlab, ylab, note,):
    # CREATE FIGURE, AXES, AND PLOT SOME DATA
    matplotlib.style.use('ggplot')
    fig, ax = plt.subplots()  # create a figure and axis (i.e. subplot) instances
    plt.subplots_adjust(left=0.25, bottom=0.25) # adjust subplot position

    # Create data
    x = data[:,0].astype(int) # range of 400 values between 0 and 4
    y_vec = data[:,1]
    y = func(y_vec, x) # define relationships between x and y

    # Draw line plot
    line, = plt.plot(x, y, lw=2, color='#3DB1FF') # draw s as a function of t
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    # DEFINE AND INITIALIZE THE DYNAMIC OBJECTS

    # Orthogonal lines at last values of x and y
    hor = plt.axhline(y[-1], c='black', lw='0.8', ls=':') # add horizontal line
    ver = plt.axvline(x[-1], c='black', lw='0.8', ls=':') # add vertical line

    # Annotate the last point (x, y)
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

    # DEFINE AND INITIALIZE SLIDER

    # Set max and min values for slide-axes [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = [x.min(), x.max(), y.min(), y.max()]

    # Create axes
    plt.axis([xmin,
              xmax + xmax * 0.05,
              ymin,
              ymax + ymax * 0.05])

    # set the slide background color
    slide_background_color = '#BDE9FF'

    # assign position for axes for slider and buttons [x, y, width, height]
    slide_ax = plt.axes([0.25, 0.12, 0.65, 0.03], axisbg=slide_background_color)
    lbut_ax = plt.axes([0.25, 0.08, 0.04, 0.03])
    rbut_ax = plt.axes([0.86, 0.08, 0.04, 0.03])


    # Create a slider and button objects at the designated axes
    slide = Slider(slide_ax, xlab, x[0].astype(int), x[-1].astype(int), valinit=x[-1].astype(int), fc='#588CBF', valfmt='%i')
    left_button = Button(lbut_ax, '<', color='#BDE8FF', hovercolor='#FFF1A8')
    right_button = Button(rbut_ax, '>', color='#BDE8FF', hovercolor='#FFF1A8')

    # DEFINE FUNCTIONS TO EXECUTE ON OBJECT EVENT (E.G. CLICK OR SLIDE)
    def update(val):
        _x = int(slide.val) # assign value of the slider to _x (the dynamic value of x)
        _y = func(y_vec,_x) # dynamic value of _y is defined my the same relationship function
        ann.xy = (_x,_y) # set annotation xy to new values
        ann.set_text('{}: {}'.format(note, np.around(_y,5))) # change text accordingly
        # change positions of straight lines
        hor.set_ydata(_y)
        ver.set_xdata(_x)
        fig.canvas.draw_idle() # draw 2D line

    def slide_plus(val):
        if slide.val + 1 <= slide.valmax:
            slide.set_val(int(slide.val + 1))
        else:
            slide.set_val(slide.valmax)

    def slide_minus(val):
        if slide.val - 1 >= slide.valmin:
            slide.set_val(int(slide.val - 1))
        else:
            slide.set_val(slide.valmin)

    # ASSIGN ACTION FUNCTIONS TO EVENTS
    slide.on_changed(update) # when the slider value is changed, call update function with the new slider position
    right_button.on_clicked(slide_plus) # on button click, call slide_plus
    left_button.on_clicked(slide_minus) # on button click, call slide_minus

    plt.show()