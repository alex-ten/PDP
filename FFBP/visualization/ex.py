import numpy as np
import matplotlib.pyplot as plt

from FFBP.visualization.hinton import hinton


A = np.random.uniform(-1, 1, size=(50, 50))
hinton(A)
plt.show()
