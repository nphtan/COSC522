import numpy as np
import timeit
import random
import math

def eucdist(a, b):
    d = 0
    if len(a) != len(b):
        return -1
    for i in range(len(a)):
        d += ((a[i] - b[i]) ** 2)
    return math.sqrt(d)

class KMeans:
    k = 0
    xdata = 0
    ydata = 0

    def __init__(self, k=6):
        self.k = k
    
    def set_k(self, k):
        self.k = k

    def set_prior(self, data):
        for i in range(0, data.shape[1]):
            x = data[:,i].reshape(self.mean.shape)
            data[:,i] = ((x - self.mean) / self.sigma).reshape(x.shape[0])

    def predict(self, xs, test_labels, seed = 1348953480572):
        random.seed(seed)
        xs = xs.T
        clusters = []
        for i in range(self.k):
            cluster = []
            for j in range(xs.shape[1]):
                cluster.append(random.random())
            clusters.append([cluster, [], []])

        iters = 0
        old_us = [[0 for x in range(xs.shape[1])] for y in range(self.k)]
        for old_u in old_us:
            print(old_u)

        while True:
            old_us = [x[0] for x in clusters]
            for cluster in clusters:
                cluster[1] = []
                cluster[2] = []
            print('Iter {}'.format(iters + 1))

            for h in range(len(xs)):
                mindex = 0
                mindist = float('inf')
                #print(pt)
                for i in range(len(clusters)):
                    d = eucdist(xs[h], clusters[i][0])
                    if d < mindist:
                        mindex = i
                        mindist = d
                clusters[mindex][1].append(list(xs[h]))
                clusters[mindex][2].append(test_labels[h])

            for cluster in clusters:
                #print(cluster[1])
                pts = [x for x in cluster[1]]
                if len(pts) == 0:
                    continue
                ave = np.mean(np.array(pts), axis=0)
                cluster[0] = list(ave)
            iters += 1
            new_us = [x[0] for x in clusters]
            if old_us == new_us:
                break
            
        for cluster in clusters:
            print(cluster[2])
