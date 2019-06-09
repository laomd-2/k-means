
import sklearn.cluster
import scipy.cluster.vq
import matplotlib.pyplot as plot
import numpy as np
import sys
import time

with open("../data-x-%s.txt" % sys.argv[1]) as fx, open("../data-y-%s.txt" % sys.argv[1]) as fy:
    data = []
    n, k = list(map(int, fx.readline().split()))
    n, k = list(map(int, fy.readline().split()))
    for i in range(n):
        data.append([float(fx.readline()), float(fy.readline())])
data = np.array(data)

start = time.time()
kmeans = sklearn.cluster.KMeans(k, max_iter=300)
labels = kmeans.fit_predict(data)
print('%.4f' % ((time.time() - start) * 1000))
# means = kmeans.cluster_centers_

# plot.scatter(data[:, 0], data[:, 1], c=labels)
# plot.scatter(means[:, 0], means[:, 1], linewidths=2)
# plot.show()