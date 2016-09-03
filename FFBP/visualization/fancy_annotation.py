import matplotlib.pyplot as plt
import numpy as np

def fancy_annotation(ax, x, y, note, l_color='black', l_width=0.8, l_style=':'):

# Orthogonal lines at last values of x and y
    hor = plt.axhline(y, c=l_color, lw=l_width, ls=l_style) # add horizontal line
    ver = plt.axvline(x, c=l_color, lw=l_width, ls=l_style) # add vertical line

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