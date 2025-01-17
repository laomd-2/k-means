import sklearn.datasets
import sklearn.cluster
import scipy.cluster.vq
import matplotlib.pyplot as plot
import sys
from math import sqrt


n = int(sys.argv[1])
m = 2
k = int(sys.argv[2])

# Generate fake data
data, labels = sklearn.datasets.make_blobs(
    n_samples=n, n_features=m, centers=k, center_box=(-20,20))

kmeans = sklearn.cluster.KMeans(k, max_iter=300)
kmeans.fit(data)
means = kmeans.cluster_centers_

with open("../data-x-%d.txt" % n, 'w') as fx, open("../data-y-%d.txt" % n, 'w') as fy:
    print(n, k, file=fx)
    print(n, k, file=fy)
    for x, y in data:
        print(x, file=fx)
        print(y, file=fy)

plot.scatter(data[:, 0], data[:, 1], c=labels)
plot.scatter(means[:, 0], means[:, 1], linewidths=2)
plot.show()