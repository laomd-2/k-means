import numpy as np
import json
import matplotlib.pyplot as plot
import sys


with open(sys.argv[1]) as fx, open(sys.argv[2]) as fy:
    data = []
    n, k = list(map(int, fx.readline().split()))
    n, k = list(map(int, fy.readline().split()))
    for i in range(n):
        data.append([float(fx.readline()), float(fy.readline())])

with open(sys.argv[3]) as f:
    n = int(f.readline())
    labels = list(map(int, f.readline().split()))
    k = int(f.readline())
    tmp = f.readline().split(' ')
    means = []
    i = 0
    while i < 2 * k:
        means.append((float(tmp[i]), float(tmp[i + 1])))
        i += 2

data = np.array(data)
means = np.array(means)
plot.scatter(data[:, 0], data[:, 1], c=labels)
plot.scatter(means[:, 0], means[:, 1], linewidths=2, marker='x', c='r')
plot.show()