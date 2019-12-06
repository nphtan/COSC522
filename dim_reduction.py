import numpy as np

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
            for i in range(0, x.shape[1]):
                diff = x[:,i].reshape((x.shape[0],1)) - mean
                sigma += diff.dot(diff.T)
            sigma = (1/x.shape[1])*sigma
            eigval,eigvec = np.linalg.eig(sigma)
        else:
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
        print(np.sum(eigval[0:n])/eigsum)
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
