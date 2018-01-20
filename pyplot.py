import numpy as np
import matplotlib.pyplot as plt

# 参考： http://blog.csdn.net/lilongsy/article/details/72903339


def f(t):
    return np.exp(-t) * np.cos(2 * np.pi * t)


t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(1)
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)
plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')

plt.figure(2)                # a second figure
# plt.subplot(211)
plt.plot(t2, np.log(t2), 'b--')

plt.show()
