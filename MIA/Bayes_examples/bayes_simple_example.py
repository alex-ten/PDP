import matplotlib
matplotlib.rcParams['backend'] = 'Agg'
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from matplotlib.widgets import Slider, Button
from MIA.Bayes_examples.dynamic_annotation import fancy_annotation
style.use('ggplot')

def infer(prior, likelihood_A, likelihood_B):
    evidence = prior * likelihood_A  +  (1 - prior) * likelihood_B
    return (prior * likelihood_A) / evidence

fig =  plt.figure(1, facecolor='w')
ax = plt.subplot(121, aspect = 'equal')  # create a figure and axis (i.e. subplot) instances
plt.subplots_adjust(left=0.25, bottom=0.25) # adjust subplot position

init_LW = 0.8 #
init_LM = 0.3
init_prior = 0.5

s = infer(np.arange(0.0,1.0,0.01), init_LW, init_LM) # define relationships of x and constants to y
l, = ax.plot(np.arange(0.0,1.0,0.01), s, lw=2, color='red') # draw s as a function of t
ax2 = plt.subplot(122, aspect = 'equal')
bar_W = ax2.bar(0, init_LW, 0.5, edgecolor='k', linewidth=2, color='red', alpha=0.4)
bar_M = ax2.bar(0.5, init_LM, 0.5, edgecolor='k', linewidth=2, color='blue', alpha=0.4)
delim = ax2.axvline(init_prior, c='k', lw=2)
ann, hline, vline = fancy_annotation(ax, init_prior, infer(init_prior, init_LW, init_LM), r'$P(W|L)$')
plt.axis([0, 1, 0, 1]) # set max and min values for axes [xmin, xmax, ymin, ymax]

axcolor = 'lightgoldenrodyellow' # assign the value for axis color to axcolor
ax_slide1 = plt.axes([0.25, 0.20, 0.65, 0.03], axisbg=axcolor) # assign position for axes for LW
ax_slide2 = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor) # assign position for axes for LM
ax_slide3 = plt.axes([0.25, 0.10, 0.65, 0.03], axisbg=axcolor) # assign position for axes for WL
slide1 = Slider(ax_slide1, r'$P(L|W)$', 0.0, 1.0, valinit=init_LW) # create a slider instance for LW using ax_slide1 from above
slide2 = Slider(ax_slide2, r'$P(L|M)$', 0.0, 1.0, valinit=init_LM) # create a slider instance for LM using ax_slide2 from above
slide3 = Slider(ax_slide3, r'$P(W)$', 0.0, 1.0, valinit=init_prior) # create a slider instance for WL using ax_slide3 from above

def update(val):
    LW = round(slide1.val,2) # assign value of LW Slider to slide1 value
    LM = round(slide2.val,2) # assign value of LM Slider to slide2 value
    WL = round(slide3.val,2) # assign value of WL Slider to slide3 value
    l.set_ydata(infer(np.arange(0.0,1.0,0.01), LW, LM)) # set y_data to l.plot.ydata (i.e. a 2D line)
    ann.xy = (WL, infer(WL,LW,LM))  # set annotation xy to new values
    ann.set_text('{}: {}'.format(r'$P(W|L)$', np.around(infer(WL, LW, LM), 5)))  # change text accordingly
    # change positions of straight lines
    hline.set_ydata(infer(WL, LW, LM))
    vline.set_xdata(WL)
    bar_W[0].set_height(LW)
    bar_W[0].set_width(WL)
    bar_M[0].set_height(LM)
    bar_M[0].set_x(WL)
    bar_M[0].set_width(1-WL)
    delim.set_xdata(WL)
    fig.canvas.draw_idle() # draws 2D line

slide1.on_changed(update) # When the slider value is changed, call update with the new slider position
slide2.on_changed(update)
slide3.on_changed(update)

resetax = plt.axes([0.8, 0.05, 0.1, 0.04]) # reserve space for Reset button
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975') # create a button instance

def reset(event):
    slide1.reset()
    slide2.reset()
    slide3.reset()
button.on_clicked(reset)

plt.show()