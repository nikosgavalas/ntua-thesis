#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(12)

x = np.arange(0.0, 50.0, 2.0)
y1 = x ** 1.3 + np.random.rand(*x.shape) * 30.0
y2 = x ** 1.5 + np.random.rand(*x.shape) * 20.0
y3 = x ** 1.2 + np.random.rand(*x.shape) * 10.0
y4 = x ** 1.7 + np.random.rand(*x.shape) * 30.0

plt.subplot(2, 1, 1)
line_1 = plt.plot(x, y1, c="green", marker='o',
                  label="circles", linestyle=(0, ()))
line_2 = plt.scatter(x, y2, c="red", marker='^', s=100,
                     label="triangles")
line_3 = plt.plot(x, y3, c="blue", marker='s',
                  label="squares", linestyle=(0, (1, 1)))
# this way only the scatter gets labeled
plt.legend([line_1, line_2, line_3], ['l1', 'l2', 'l3'], loc='upper left')
plt.ylabel("y-axis up")


plt.subplot(2, 1, 2)
line_4 = plt.plot(x, y4, c="black", marker='D',
                  label="diamond", linestyle=(0, (5, 5)))

plt.xlabel("x-axis")
plt.ylabel("y-axis down")
# this way only the normal plot gets labeled
plt.legend(loc='upper left')

plt.show()
