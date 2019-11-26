import csv
import numpy as np
import timeit

def filter_retweets(data):
    no_rt = []
    for sample in data:
        tweet = sample[1]
        if tweet.count('RT') == 0:
            no_rt.append(sample)
    return no_rt

def extract_features(data):
    features = np.zeros((5,len(data)))
    for i in range(0,len(data)):
        tweet = data[i][1]
        upper = 0
        for word in tweet.split():
            if word.isupper():
                upper += 1
        features[0,i] = tweet.count('!')
#        features[1,i] = tweet.lower().count('collusion')
        features[1,i] = tweet.lower().count('collusion') + tweet.lower().count('fake news') + tweet.lower().count('quid pro quo') + tweet.lower().count('russia')
        features[2,i] = tweet.count('@')
        features[3,i] = upper
        features[4,i] = tweet.lower().count('maga') + tweet.lower().count('make america great again') + tweet.lower().count('#makeamericagreatagain') + tweet.lower().count('make #americagreatagain')
#        features[5,i] = tweet.lower().count('great')
    return features

def standardize(data, mean, sigma):
    for i in range(0, data.shape[1]):
        x = data[:,i].reshape(mean.shape)
        data[:,i] = ((x-mean)/sigma).reshape(x.shape[0])

def perf_eval(predict, true):
    num_samples = predict.shape[0]
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    for i in range(0, num_samples):
        if predict[i] == 0:
            if predict[i] == true[i]:
                tn += 1
            else:
                fn += 1
        else:
            if predict[i] == true[i]:
                tp += 1
            else:
                fp += 1
    return (tp,tn,fn,fp)

class PCA:
    w = ''
    tol = 0
    eigenvalues = ''
    eigenvectors = ''
    mean = ''
    decomp = ''

    def __init__(self, decomp='eig'):
        self.w = 0
        self.tol = 0
        self.decomp = decomp

    def reset(self):
        self.w = 0
        self.tol = 0

    def get_w(self):
        return self.w

    def get_tol(self):
        return self.tol

    def setup(self, x, tol):
        self.tol = tol
        sigma = np.zeros((x.shape[0],x.shape[0]))
        mean = np.mean(x, axis=1).reshape((x.shape[0], 1))
        self.mean = mean
        eigval = 0
        eigvec = 0
        if self.decomp == 'eig':
            print('PCA: using eigenvalue decomposition')
            for i in range(0, x.shape[1]):
                diff = x[:,i].reshape((x.shape[0],1)) - mean
                sigma += diff.dot(diff.T)
            sigma = (1/x.shape[1])*sigma
            eigval,eigvec = np.linalg.eig(sigma)
        else:
            print('PCA: using SVD')
            u,s,vt = np.linalg.svd(x.T,full_matrices=False)
            eigval = s*s/(x.shape[1]-1)
            eigvec = vt.T
        eigsum = np.sum(eigval)
        order = np.argsort(eigval)[::-1]
        eigvec = eigvec[:,order]
        eigval = eigval[order]
        self.eigenvalues = eigval
        self.eigenvectors = eigvec
        n = x.shape[0]
        while np.sum(eigval[0:n])/eigsum > self.tol:
            n -= 1
        n+=1
        self.w = eigvec[:,0:n]

    def reduce(self, data):
        return self.w.T.dot(data)

class FLD:
    w = ''
    sigma = ''
    mean0 = 0
    mean1 = 0
    s0 = 0
    s1 = 0

    def __init__(self):
        self.w = 0

    def reset(self):
        self.w = 0

    def setup(self, x, xclass):
        num_var = x.shape[0]
        samples0 = np.array([x[:,i] for i in range(0, x.shape[1]) if xclass[i] == 0]).T
        samples1 = np.array([x[:,i] for i in range(0, x.shape[1]) if xclass[i] == 1]).T
        mean0 = np.mean(samples0, axis=1).reshape((num_var, 1))
        mean1 = np.mean(samples1, axis=1).reshape((num_var, 1))
        self.mean0 = mean0
        self.mean1 = mean1
        S0 = np.zeros((num_var, num_var))
        S1 = np.zeros((num_var, num_var))
        for i in range(0, samples0.shape[1]):
            diff = samples0[:,i].reshape((num_var,1)) - mean0
            S0 = np.add(np.dot(diff, diff.T), S0)
        for i in range(0, samples1.shape[1]):
            diff = samples1[:,i].reshape((num_var,1)) - mean1
            S1 = np.add(np.dot(diff, diff.T), S1)
        self.s0 = S0
        self.s1 = S1
        Sw = S0 + S1
        self.sigma = Sw
        self.w = np.linalg.inv(Sw).dot(mean0-mean1)

    def reduce(self, data):
        return self.w.T.dot(data)

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

    def predict(self, xtest):
        start = timeit.default_timer()
        ytest = np.zeros(xtest.shape[1])
        # For each test sample
        for i in range(0, xtest.shape[1]):
            x = xtest[:,i]
            distance = np.zeros((self.xdata.shape[1],1))
            # Compute distance from each training sample to test sample
            for j in range(0, self.xdata.shape[1]):
                distance[j] = np.linalg.norm(x-self.xdata[:,j], ord=2)
            # Find k nearest neighbors
            nearest = distance.argsort(axis=0)[:self.k]
            num0 = 0
            num1 = 0
            # Compute which class based on number of neighbors
            for a in range(0,self.k):
                if self.ydata[nearest[a]] == 0:
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
        stop = timeit.default_timer()
        print('KNN run time: ', stop-start, 's')
        return ytest

def main():

    data = []
    test_data = []
    with open('ObTr1.csv', 'r', encoding='utf8', errors='ignore') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(row)
    with open('trump_hillary_tweets.csv', 'r', encoding='utf8', errors='ignore') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            test_data.append(row)
    
    data.pop(0)
    test_data.pop(0)

    full_data = data
    no_rt_data = filter_retweets(data)

    test_labels = np.ones(len(test_data))
    for i in range(0,len(test_data)):
        test_labels[i] = test_data[i][2]
    true_labels = np.ones(len(data))
    for i in range(0,len(data)):
        true_labels[i] = data[i][2]
    no_rt_labels = np.ones(len(no_rt_data))
    for i in range(0,len(no_rt_data)):
        no_rt_labels[i] = no_rt_data[i][2]
    
    features = extract_features(data)
    no_rt_features = extract_features(no_rt_data)
    test_features = extract_features(test_data)
    test_features2 = test_features

    mean = np.mean(features, axis=1).reshape((features.shape[0],1))
    sigma = np.std(features, axis=1).reshape((features.shape[0],1))
    mean2 = np.mean(no_rt_features, axis=1).reshape((no_rt_features.shape[0],1))
    sigma2 = np.std(no_rt_features, axis=1).reshape((no_rt_features.shape[0],1))
    standardize(features, mean, sigma)
    standardize(no_rt_features, mean2, sigma2)
    standardize(test_features, mean, sigma)
    standardize(test_features2, mean2, sigma2)

#    fld = FLD()
#    fld.setup(features, true_labels)
#    features = fld.reduce(features)
#    test_features = fld.reduce(test_features)
#
#    fld2 = FLD()
#    fld2.setup(no_rt_features, no_rt_labels)
#    no_rt_features = fld.reduce(no_rt_features)
#    test_features2 = fld.reduce(test_features2)

    pca = PCA()
    pca.setup(features, 0.8)
    features = pca.reduce(features)
    test_features = pca.reduce(test_features)

    pca2 = PCA()
    pca2.setup(no_rt_features, 0.8)
    no_rt_features = pca.reduce(no_rt_features)
    test_features2 = pca.reduce(test_features2)

    k = 3
    print(k)
    knn_model = KNN(k)
    knn_model.fit(features, true_labels)
    ymodel = knn_model.predict(test_features)
    tp,tn,fn,fp = perf_eval(ymodel, test_labels)
    print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
    knn_model2 = KNN(k)
    knn_model2.fit(no_rt_features, no_rt_labels)
    ymodel = knn_model2.predict(test_features2)
    tp,tn,fn,fp = perf_eval(ymodel, test_labels)
    print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))

if __name__ == "__main__":
    main()
