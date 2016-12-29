import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def fancy_annotation(ax, x, y, note, l_color='black', l_width=0.8, l_style=':'):

# Orthogonal lines at last values of x and y
    hor = ax.axhline(y, c=l_color, lw=l_width, ls=l_style) # add horizontal line
    ver = ax.axvline(x, c=l_color, lw=l_width, ls=l_style) # add vertical line

    # Annotate the last point (x, y)
    ann = ax.annotate('{}: {}'.format(note, np.around(y,5)),  # annotation string
                      # annotation coordinates:
                      xy=(x, y),  # position of element to annotate
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
    return ann, hor, ver

def fancy_line(y, xlab, ylab, note):
    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.25, bottom=0.25, top=0.70, right=0.75)  # adjust subplot position
    ann, hline, vline = fancy_annotation(ax, y.index(y[-1]), y[-1], note)
    return

def error_figure(data, xlab, ylab, note):
    # CREATE AXES, AND PLOT SOME DATA
    plt.clf()
    ax = plt.subplot(111)
    plt.subplots_adjust(left = 0.25, bottom = 0.25, top = 0.70, right = 0.75 ) # adjust subplot position

    # Arange data for internal use
    y = np.array(data, dtype=float)
    x = np.arange(len(y), dtype=int)

    # Draw line plot
    plt.plot(x, y, lw=2, color='#3DB1FF') # plot y as a function of x
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    # DEFINE AND INITIALIZE THE DYNAMIC OBJECTS

    ann, hline, vline = fancy_annotation(ax, x[-1], y[-1], note)

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

    # Create a slider and button objects on designated axes
    slide = Slider(slide_ax, xlab, x[0].astype(int), x[-1].astype(int), valinit=x[-1].astype(int), fc='#588CBF', valfmt='%i')
    left_button = Button(lbut_ax, '<', color='#BDE8FF', hovercolor='#FFF1A8')
    right_button = Button(rbut_ax, '>', color='#BDE8FF', hovercolor='#FFF1A8')

    # DEFINE FUNCTIONS TO EXECUTE ON OBJECT EVENT (E.G. CLICK OR SLIDE)
    def drag_slider(val):
        _x = int(slide.val) # assign value of the slider to _x (the dynamic value of x)
        _y = y[_x] # dynamic value of _y is defined my the same relationship function
        ann.xy = (_x,_y) # set annotation xy to new values
        ann.set_text('{}: {}'.format(note, np.around(_y,5))) # change text accordingly
        # change positions of straight lines
        hline.set_ydata(_y)
        vline.set_xdata(_x)

    def slide_plus(val):
        print('RIGHT BUTTON WAS')
        if slide.val + 1 <= slide.valmax:
            slide.set_val(int(slide.val + 1))
        else:
            slide.set_val(slide.valmax)

    def slide_minus(val):
        print('LEFT BUTTON WAS')
        if slide.val - 1 >= slide.valmin:
            slide.set_val(int(slide.val - 1))
        else:
            slide.set_val(slide.valmin)


    # PASS FUNCTIONS TO EVENTS
    slide.on_changed(drag_slider) # when the slider value is changed, call update function with the new slider position
    right_button.on_clicked(slide_plus) # on button click, call slide_plus
    left_button.on_clicked(slide_minus) # on button click, call slide_minus


def demo():
    print("Here's a demo")
    fig = plt.figure(1)
    data = [10,9,9,8,7]
    error_figure(fig, data, 'x','y', 'f(x)')
    input('hit enter to update')
    data = data + [8,6,6,5,4,3,2]
    error_figure(fig, data, 'x','y', 'f(x)')
    input('hit enter to quit')

if __name__=='__main__': demo()