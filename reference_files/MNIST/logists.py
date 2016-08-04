import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-4.0,4.0,0.1)
a = [3.0,1.0,0.2]

def softmax(a):
    return (np.exp(a) / np.sum(np.exp(a)))

def logit(a):
    return (1 / (1 + np.exp(-a)))

print(softmax(a))
print(softmax([0.03,0.010,0.002]))

# x_soft = softmax(x)
# print(np.around(x_soft,1))
# x_log = logit(x)
#
# plt.subplot(221)
# plt.title('SOFTMAX')
# plt.ylim(-1,1)
# plt.plot(x, x_soft, lw=2)
#
# plt.subplot(222)
# plt.title('LOGIT')
# plt.ylim(-1,1)
# plt.plot(x, x_log, lw=2)
# plt.subplot(212)
#
# plt.plot(x, x_soft, lw=3)
# plt.plot(x, x_log, lw=3)
# plt.ylim(-1,1)
# plt.title('BOTH')
# plt.show()