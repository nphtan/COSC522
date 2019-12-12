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

    def predict(self, xs, test_labels, seed = 2):
        random.seed(seed)
        xs = xs.T
        clusters = []
        preds = [0 for x in range(len(xs))]
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
                preds[h] = mindex

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
        
        num0s = float(len([x for x in test_labels if x == 0]))
        num1s = float(len([x for x in test_labels if x == 1]))
        finals = zip(xs, test_labels)
        y = [x[0] for x in finals if x[1] == 1.0]
        mean1 = np.mean(y, axis=0)
        finals = zip(xs, test_labels)
        y = [x[0] for x in finals if x[1] == 0.0]
        mean0 = np.mean(y, axis=0)
        tpr = 0.0
        tnr = 0.0
        fpr = 0.0
        fnr = 0.0
        for i in range(len(clusters)):
            pred0 = len([x for x in clusters[i][2] if x == 0])
            pred1 = len([x for x in clusters[i][2] if x == 1])
            if eucdist(clusters[i][0], mean0) < eucdist(clusters[i][0], mean1):
                print('Cluster {} assigned to 0'.format(i))
                assigned = 0
                tnr = pred0 / num0s
                fpr = 1 - tnr
            else:
                print('Cluster {} assigned to 1'.format(i))
                assigned = 1
                tpr = pred1 / num1s
                fnr = 1 - tpr
            #print('Cluster {}: 0 = {}/{}'.format(i, pred0, (pred0 + pred1)))
            #print('Cluster {}: 1 = {}/{}'.format(i, pred1, (pred0 + pred1)))

        print('TPR: {}'.format(tpr))
        print('TNR: {}'.format(tnr))
        print('FPR: {}'.format(fpr))
        print('FNR: {}'.format(fnr))
#        print(preds)
        print(len(preds), len(xs))
        return preds


