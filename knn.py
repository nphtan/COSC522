import numpy as np
import timeit

class KNN:
    k = 0
    xdata = 0
    ydata = 0
    mean = 0
    sigma = 0
    prior0 = -1
    prior1 = -1
    class0percent = -1
    class1percent = -1
    distances = []
    xtest = []

    def __init__(self, k, prior0 = -1, prior1 = -1):
        self.k = k
        self.prior0 = prior0
        self.prior1 = prior1

    def set_k(self, k):
        self.k = k

    def set_prior(self, p0, p1):
        self.prior0 = p0
        self.prior1 = p1

    def standardize(self, data):
        for i in range(0, data.shape[1]):
            x = data[:,i].reshape(self.mean.shape)
            data[:,i] = ((x-self.mean)/self.sigma).reshape(x.shape[0])

    def fit(self, xtrain, ytrain):
        self.ydata = ytrain
        self.xdata = xtrain
        samples0 = np.array([xtrain[:,i] for i in range(0, xtrain.shape[1]) if ytrain[i] == 0]).T
        samples1 = np.array([xtrain[:,i] for i in range(0, xtrain.shape[1]) if ytrain[i] == 1]).T
        # Save N_k/N for different prior probabilities
        self.class0percent = samples0.shape[0]/(samples0.shape[0]+samples1.shape[0])
        self.class1percent = samples1.shape[0]/(samples0.shape[0]+samples1.shape[0])

    def predict(self, xtest, norm=2):
        self.xtest = xtest
        ytest = np.zeros(xtest.shape[1])
        # For each test sample
        distance = np.zeros((xtest.shape[1], self.xdata.shape[1]))
        for i in range(0, xtest.shape[1]):
            x = xtest[:,i]
            # Compute distance from each training sample to test sample
            for j in range(0, self.xdata.shape[1]):
                distance[i,j] = np.linalg.norm(x-self.xdata[:,j], ord=2)
            # Find k nearest neighbors
            nearest = distance[i,:].reshape((self.xdata.shape[1],1)).argsort(axis=0)[:self.k]
            num0 = 0
            num1 = 0
            # Compute which class based on number of neighbors
            for a in range(0,self.k):
                if self.ydata[int(nearest[a])] == 0:
                    num0 += 1
                else:
                    num1 += 1
            if self.prior0 == -1:
                if num0 == num1:
                    ytest[i] = self.ydata[nearest[0]]
                elif num0 > num1:
                    ytest[i] = 0
                else:
                    ytest[i] = 1
            else:
                # Calculate a-priori probability for user set prior probabilities
                p0 = self.prior0*(num0*self.k)/self.class0percent
                p1 = self.prior1*(num1*self.k)/self.class1percent
                if p0 == p1:
                    ytest[i] = self.ydata[nearest[0]]
                elif p0 > p1:
                    ytest[i] = 0
                else:
                    ytest[i] = 1
        self.distances = distance
        return ytest
    
    def predict_prob(self, xtest, norm=2):
        self.xtest = xtest
        ytest = np.zeros(xtest.shape[1])
        probs = np.zeros((xtest.shape[1],2), dtype=np.float64)
        distance = np.zeros((xtest.shape[1], self.xdata.shape[1]))
        # For each test sample
        for i in range(0, xtest.shape[1]):
            x = xtest[:,i]
            if self.distances == []:
                # Compute distance from each training sample to test sample
                for j in range(0, self.xdata.shape[1]):
                    distance[i,j] = np.linalg.norm(x-self.xdata[:,j], ord=2)
            else:
                distance = self.distances
            # Find k nearest neighbors
            nearest = distance[i,:].reshape((self.xdata.shape[1],1)).argsort(axis=0)[:self.k]
            num0 = 0
            num1 = 0
            for a in range(0,self.k):
                if self.ydata[int(nearest[a])] == 0:
                    num0 += 1
                else:
                    num1 += 1
            probs[i,0] = num0/self.k
            probs[i,1] = num1/self.k
        return probs

    def predict_k(self, k):
        ytest = np.zeros(self.xtest.shape[1])
        distance = np.zeros((self.xtest.shape[1], self.xdata.shape[1]))
        for i in range(0, self.xtest.shape[1]):
            if self.distances == []:
                print('Need to run predict first')
                x = self.xtest[:,i]
                # Compute distance from each training sample to test sample
                for j in range(0, self.xdata.shape[1]):
                    distance[i,j] = np.linalg.norm(x-self.xdata[:,j], ord=2)
                self.distances = distance

            distance = self.distances

            # Find k nearest neighbors
            nearest = distance[i,:].reshape((self.xdata.shape[1],1)).argsort(axis=0)[:k]
            num0 = 0
            num1 = 0
            # Compute which class based on number of neighbors
            for a in range(0,k):
                if self.ydata[int(nearest[a])] == 0:
                    num0 += 1
                else:
                    num1 += 1
            if self.prior0 == -1:
                if num0 == num1:
                    ytest[i] = self.ydata[nearest[0]]
                elif num0 > num1:
                    ytest[i] = 0
                else:
                    ytest[i] = 1
            else:
                # Calculate a-priori probability for user set prior probabilities
                p0 = self.prior0*(num0*k)/self.class0percent
                p1 = self.prior1*(num1*k)/self.class1percent
                if p0 == p1:
                    ytest[i] = self.ydata[nearest[0]]
                elif p0 > p1:
                    ytest[i] = 0
                else:
                    ytest[i] = 1
        return ytest

