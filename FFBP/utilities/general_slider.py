import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

fig, ax = plt.subplots()  # create a figure and axis (i.e. subplot) instances
plt.subplots_adjust(left=0.25, bottom=0.25) # adjust subplot position
t = np.arange(0.0, 1.0, 0.001) # range of 1000 values between 0 and 1
a0 = 5
s = a0*np.sin(2*np.pi*t) # define relationships of x and constants to y
l, = plt.plot(t, s, lw=2, color='red') # draw s as a function of t
plt.axis([0, 1, -10, 10]) # set max and min values for axes [xmin, xmax, ymin, ymax]

axcolor = 'lightgoldenrodyellow' # assign the value for axis color to axcolor
axamp = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor) # assign position for axes for AMPLITUDE SLIDER

samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0) # create a slider instance for AMPLITUDE using axamp from above


def update(val):
    amp = samp.val # assign value of samp Slider to amp
    l.set_ydata(amp*np.sin(2*np.pi*t)) # set y_data to l.plot.ydata (i.e. a 2D line)
    fig.canvas.draw_idle() # draws 2D line
samp.on_changed(update) # When the slider value is changed, call update with the new slider position

resetax = plt.axes([0.8, 0.025, 0.1, 0.04]) # reserve space for Reset button
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975') # create a button instance


def reset(event):
    samp.reset()
button.on_clicked(reset)

rax = plt.axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


def colorfunc(label):
    l.set_color(label)
    fig.canvas.draw_idle()
radio.on_clicked(colorfunc)

plt.show()