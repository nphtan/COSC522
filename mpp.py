import math
import numpy as np
import time

def gaussian(xy, mu, sigma):
    pi = math.pi
    inv_sigma = np.linalg.inv(sigma)
    det_sigma = math.sqrt(np.linalg.det(sigma))
    z = np.zeros((xy.shape[1], 1))
    for i in range(0, xy.shape[1]):
        x_mu = xy[:,i].reshape((2,1))-mu
        z[i] = 1/((2*pi)*(det_sigma)) * math.exp(-0.5* x_mu.T.dot(inv_sigma).dot(x_mu))
    return z

class MPP:
    case = -1
    prior_prob0 = 0.5
    prior_prob1 = 0.5
    bimodal_weight0a = 0.5
    bimodal_weight0b = 0.5
    bimodal_weight1a = 0.5
    bimodal_weight1b = 0.5
    num_features = 0
    mu0 = np.zeros((2,1))
    mu1 = np.zeros((2,1))
    mu0a = np.zeros((2,1))
    mu0b = np.zeros((2,1))
    mu1a = np.zeros((2,1))
    mu1b = np.zeros((2,1))
    sigma0 = np.zeros((2,2))
    sigma1 = np.zeros((2,2))
    sigma0a = np.zeros((2,2))
    sigma0b = np.zeros((2,2))
    sigma1a = np.zeros((2,2))
    sigma1b = np.zeros((2,2))

    def __init__(self, c):
        self.case = int(c)

    def set_prior(self, a, b):
        self.prior_prob0 = a
        self.prior_prob1 = b

    def set_bimodal_weights(self, b0a, b0b, b1a, b1b):
        self.bimodal_weight0a = b0a
        self.bimodal_weight0b = b0b
        self.bimodal_weight1a = b1a
        self.bimodal_weight1b = b1b

    def fit(self, xtrain, ytrain):
        self.num_features = xtrain.shape[0]
        self.mu0 = np.zeros((self.num_features,1))
        self.mu1 = np.zeros((self.num_features,1))
        self.mu0a = np.zeros((self.num_features,1))
        self.mu0b = np.zeros((self.num_features,1))
        self.mu1a = np.zeros((self.num_features,1))
        self.mu1b = np.zeros((self.num_features,1))
        self.sigma0 = np.zeros((self.num_features,self.num_features))
        self.sigma1 = np.zeros((self.num_features,self.num_features))
        self.sigma0a = np.zeros((self.num_features,self.num_features))
        self.sigma0b = np.zeros((self.num_features,self.num_features))
        self.sigma1a = np.zeros((self.num_features,self.num_features))
        self.sigma1b = np.zeros((self.num_features,self.num_features))
        sums = np.zeros((self.num_features,2))
        num1 = np.count_nonzero(ytrain)
        num0 = ytrain.shape[0] - num1
        for i in range(0,num0):
            sums[:,0] += xtrain[:,i]
        for i in range(0,num1):
            sums[:,1] += xtrain[:,num0+i]
#        for i in range(0, num0):
#            sumx[0] += xtrain[0,i]
#            sumy[0] += xtrain[1,i]
#        for i in range(0, num1):
#            sumx[1] += xtrain[0, num0+i]
#            sumy[1] += xtrain[1, num0+i]
        # Compute mean for each class
        self.mu0[:] = sums[:,0].reshape((self.num_features,1))/num0
        self.mu1[:] = sums[:,1].reshape((self.num_features,1))/num0

#        self.mu0[0] = sumx[0]/num0
#        self.mu0[1] = sumy[0]/num0
#        self.mu1[0] = sumx[1]/num0
#        self.mu1[1] = sumy[1]/num0

        if self.case == 1:
            sigma = np.var(xtrain)
            self.sigma0 = np.identity(2)*(sigma)
            self.sigma1 = self.sigma0
        elif self.case == 2:
            for i in range(0, num0):
                diff = xtrain[:,i].reshape((self.num_features,1)) - self.mu0
                self.sigma0 += np.dot(diff, np.transpose(diff))
            self.sigma0 = self.sigma0*(1/(num0-1))
            self.sigma1 = self.sigma0
        elif self.case == 3:
            for i in range(0, num0):
                diff = xtrain[:,i].reshape((self.num_features,1)) - self.mu0
                self.sigma0 += np.dot(diff, np.transpose(diff))
            for i in range(num0, ytrain.shape[0]):
                diff = xtrain[:,i].reshape((self.num_features,1)) - self.mu1
                self.sigma1 += np.dot(diff, np.transpose(diff))
            self.sigma0 = self.sigma0*(1/(num0-1))
            self.sigma1 = self.sigma1*(1/(num1-1))
        elif self.case == 4:
            #Compute mean mu with split at x = -0.25 for class 0
            #Compute mean mu with split at x = 0.0 for class 1
            class0_samplesA = np.array([(xtrain[0,i], xtrain[1,i]) for i in range(0, (num0+num1)) if xtrain[0,i] <= -0.25 and ytrain[i] == 0])
            class0_samplesB = np.array([(xtrain[0,i], xtrain[1,i]) for i in range(0, (num0+num1)) if xtrain[0,i] > -0.25 and ytrain[i] == 0])
            class1_samplesA = np.array([(xtrain[0,i], xtrain[1,i]) for i in range(0, (num0+num1)) if xtrain[0,i] <= 0 and ytrain[i] == 1])
            class1_samplesB = np.array([(xtrain[0,i], xtrain[1,i]) for i in range(0, (num0+num1)) if xtrain[0,i] > 0 and ytrain[i] == 1])
            self.mu0a = np.mean(class0_samplesA, axis=0).reshape((self.num_features,1))
            self.mu0b = np.mean(class0_samplesB, axis=0).reshape((self.num_features,1))
            self.mu1a = np.mean(class1_samplesA, axis=0).reshape((self.num_features,1))
            self.mu1b = np.mean(class1_samplesB, axis=0).reshape((self.num_features,1))
            num0a = class0_samplesA.shape[0]
            num0b = class0_samplesB.shape[0]
            num1a = class1_samplesA.shape[0]
            num1b = class1_samplesB.shape[0]
            for i in range(0, num0a):
                diff = class0_samplesA[i,:].reshape((self.num_features,1)) - self.mu0a
                self.sigma0a += np.dot(diff, np.transpose(diff))
            self.sigma0a = self.sigma0a*(1/(num0a-1))
            for i in range(0, num0b):
                diff = class0_samplesB[i,:].reshape((self.num_features,1)) - self.mu0b
                self.sigma0b += np.dot(diff, np.transpose(diff))
            self.sigma0b = self.sigma0b*(1/(num0b-1))
            for i in range(0, num1a):
                diff = class1_samplesA[i,:].reshape((self.num_features,1)) - self.mu1a
                self.sigma1a += np.dot(diff, np.transpose(diff))
            self.sigma1a = self.sigma1a*(1/(num1a-1))
            for i in range(0, num1b):
                diff = class1_samplesB[i,:].reshape((self.num_features,1)) - self.mu1b
                self.sigma1b += np.dot(diff, np.transpose(diff))
            self.sigma1b = self.sigma1b*(1/(num1b-1))

    def predict(self, xtest):
        ytest = np.zeros( (xtest.shape[1], 1) )
        if self.case == 1:
            for i in range(0, xtest.shape[1]):
                diff0 = np.zeros((self.num_features,1))
                diff0 = xtest[:,i].reshape((self.num_features,1)) - self.mu0
                diff1 = np.zeros((self.num_features,1))
                diff1 = xtest[:,i].reshape((self.num_features,1)) - self.mu1
                x = np.zeros((self.num_features,1))
                x = xtest[:,i].reshape((self.num_features,1))
                sigma = self.sigma0[0,0]
                inv_sigma = 1/(sigma**2)
                ln_prior0 = math.log(self.prior_prob0)
                ln_prior1 = math.log(self.prior_prob1)
                prob0 = inv_sigma*self.mu0.T.dot(x) \
                        - 0.5*inv_sigma*self.mu0.T.dot(self.mu0) \
                        + ln_prior0
                prob1 = inv_sigma*self.mu1.T.dot(x) \
                        - 0.5*inv_sigma*self.mu1.T.dot(self.mu1) \
                        + ln_prior1
                if prob0 > prob1:
                    ytest[i] = 0
                else:
                    ytest[i] = 1
        elif self.case == 2:
            for i in range(0, xtest.shape[1]):
                x = xtest[:,i]
                diff0 = np.zeros((self.num_features,1))
                diff1 = np.zeros((self.num_features,1))
                diff0 = xtest[:,i].reshape((self.num_features,1)) - self.mu0
                diff1 = xtest[:,i].reshape((self.num_features,1)) - self.mu1
                inv_sigma0 = np.linalg.inv(self.sigma0)
                inv_sigma1 = np.linalg.inv(self.sigma1)
                ln_prior0 = math.log(self.prior_prob0)
                ln_prior1 = math.log(self.prior_prob1)
                prob0 = -0.5*(diff0.T.dot(inv_sigma0).dot(diff0)) + ln_prior0
                prob1 = -0.5*(diff1.T.dot(inv_sigma1).dot(diff1)) + ln_prior1
                if prob0 > prob1:
                    ytest[i] = 0
                else:
                    ytest[i] = 1
        elif self.case == 3:
            for i in range(0, xtest.shape[1]):
                x = np.zeros((self.num_features,1))
                x = xtest[:,i].reshape((self.num_features,1))
                diff0 = np.zeros((self.num_features,1))
                diff0 = xtest[:,i].reshape((self.num_features,1)) - self.mu0
                diff1 = np.zeros((self.num_features,1))
                diff1 = xtest[:,i].reshape((self.num_features,1)) - self.mu1
                inv_sigma0 = np.linalg.inv(self.sigma0)
                inv_sigma1 = np.linalg.inv(self.sigma1)
                det_sigma0 = np.linalg.det(self.sigma0)
                det_sigma1 = np.linalg.det(self.sigma1)
                ln_prior0 = math.log(self.prior_prob0)
                ln_prior1 = math.log(self.prior_prob1)
                prob0 = -0.5*(diff0.T.dot(inv_sigma0).dot(diff0)) \
                        -0.5*det_sigma0 + ln_prior0
                prob1 = -0.5*(diff1.T.dot(inv_sigma1).dot(diff1)) \
                        -0.5*det_sigma1 + ln_prior1
                if prob0 > prob1:
                    ytest[i] = 0
                else:
                    ytest[i] = 1
        elif self.case == 4:
            prob0 = self.bimodal_weight0a*gaussian(xtest, self.mu0a, self.sigma0a) + \
                    self.bimodal_weight0b*gaussian(xtest, self.mu0b, self.sigma0b) + \
                    self.prior_prob0
            prob1 = self.bimodal_weight1a*gaussian(xtest, self.mu1a, self.sigma1a) + \
                    self.bimodal_weight1b*gaussian(xtest, self.mu1b, self.sigma1b) + \
                    self.prior_prob1
            for i in range(0, xtest.shape[1]):
                if prob0[i] > prob1[i]:
                    ytest[i] = 0
                else:
                    ytest[i] = 1
        return ytest

    def predict_prob(self, xtest):
        probs = np.zeros((xtest.shape[1],2), dtype=np.float64)
        if self.case == 1:
            for i in range(0, xtest.shape[1]):
                diff0 = np.zeros((self.num_features,1))
                diff0 = xtest[:,i].reshape((self.num_features,1)) - self.mu0
                diff1 = np.zeros((self.num_features,1))
                diff1 = xtest[:,i].reshape((self.num_features,1)) - self.mu1
                x = np.zeros((self.num_features,1))
                x = xtest[:,i].reshape((self.num_features,1))
                sigma = self.sigma0[0,0]
                inv_sigma = 1/(sigma**2)
                ln_prior0 = math.log(self.prior_prob0)
                ln_prior1 = math.log(self.prior_prob1)
                prob0 = inv_sigma*self.mu0.T.dot(x) \
                        - 0.5*inv_sigma*self.mu0.T.dot(self.mu0) \
                        + ln_prior0
                prob1 = inv_sigma*self.mu1.T.dot(x) \
                        - 0.5*inv_sigma*self.mu1.T.dot(self.mu1) \
                        + ln_prior1
                probs[i,0] = math.exp(prob0)/(math.exp(prob0)+math.exp(prob1))
                probs[i,1] = math.exp(prob1)/(math.exp(prob0)+math.exp(prob1))
        elif self.case == 2:
            for i in range(0, xtest.shape[1]):
                x = xtest[:,i]
                diff0 = np.zeros((self.num_features,1))
                diff1 = np.zeros((self.num_features,1))
                diff0 = xtest[:,i].reshape((self.num_features,1)) - self.mu0
                diff1 = xtest[:,i].reshape((self.num_features,1)) - self.mu1
                inv_sigma0 = np.linalg.inv(self.sigma0)
                inv_sigma1 = np.linalg.inv(self.sigma1)
                ln_prior0 = math.log(self.prior_prob0)
                ln_prior1 = math.log(self.prior_prob1)
                prob0 = -0.5*(diff0.T.dot(inv_sigma0).dot(diff0)) + ln_prior0
                prob1 = -0.5*(diff1.T.dot(inv_sigma1).dot(diff1)) + ln_prior1
                probs[i,0] = math.exp(prob0)/(math.exp(prob0)+math.exp(prob1))
                probs[i,1] = math.exp(prob1)/(math.exp(prob0)+math.exp(prob1))
        elif self.case == 3:
            for i in range(0, xtest.shape[1]):
                x = np.zeros((self.num_features,1))
                x = xtest[:,i].reshape((self.num_features,1))
                diff0 = np.zeros((self.num_features,1))
                diff0 = xtest[:,i].reshape((self.num_features,1)) - self.mu0
                diff1 = np.zeros((self.num_features,1))
                diff1 = xtest[:,i].reshape((self.num_features,1)) - self.mu1
                inv_sigma0 = np.linalg.inv(self.sigma0)
                inv_sigma1 = np.linalg.inv(self.sigma1)
                det_sigma0 = np.linalg.det(self.sigma0)
                det_sigma1 = np.linalg.det(self.sigma1)
                ln_prior0 = math.log(self.prior_prob0)
                ln_prior1 = math.log(self.prior_prob1)
                prob0 = -0.5*(diff0.T.dot(inv_sigma0).dot(diff0)) \
                        -0.5*det_sigma0 + ln_prior0
                prob1 = -0.5*(diff1.T.dot(inv_sigma1).dot(diff1)) \
                        -0.5*det_sigma1 + ln_prior1
                probs[i,0] = math.exp(prob0)/(math.exp(prob0)+math.exp(prob1))
                probs[i,1] = math.exp(prob1)/(math.exp(prob0)+math.exp(prob1))
        elif self.case == 4:
            prob0 = self.bimodal_weight0a*gaussian(xtest, self.mu0a, self.sigma0a) + \
                    self.bimodal_weight0b*gaussian(xtest, self.mu0b, self.sigma0b) + \
                    self.prior_prob0
            prob1 = self.bimodal_weight1a*gaussian(xtest, self.mu1a, self.sigma1a) + \
                    self.bimodal_weight1b*gaussian(xtest, self.mu1b, self.sigma1b) + \
                    self.prior_prob1
            for i in range(0, xtest.shape[1]):
                probs[i,0] = math.exp(prob0)/(math.exp(prob0)+math.exp(prob1))
                probs[i,1] = math.exp(prob1)/(math.exp(prob0)+math.exp(prob1))
        return probs

class Model:
    def case1(self, t):
        # Setup the points matrix and averages in a numpy-friendly format
        pts = [np.matrix((x[0], x[1])).T for x in t]
        u_g0 = np.matrix((self.ux_0, self.uy_0)).T
        u_g1 = np.matrix((self.ux_1, self.uy_1)).T
        ps = []
        v = self.v_0

        # Calculate gi(x) for each class and make a prediction
        for x in pts:
            finals = []
            for u, p in [(u_g0, self.p_0), (u_g1, self.p_1)]:
                partial1 = np.matmul((u.T / v), x)
                partial2 = np.matmul(u.T, u) / (2 * v)
                finals.append(float(partial1 - partial2 + np.log(p)))
            if finals[0] > finals[1]:
                ps.append(0)
            else:
                ps.append(1)
        return ps

    def case2(self, t):
        # Set up the points matrix, averages, and covariance matrix in a
        # numpy-friendly format
        pts = [np.matrix((x[0], x[1])).T for x in t]
        u_g0 = np.matrix((self.ux_0, self.uy_0)).T
        u_g1 = np.matrix((self.ux_1, self.uy_1)).T
        cvm = self.cv_0
        ps = []
        
        # Predict each point
        for x in pts:
            finals = []
            for u, p in [(u_g0, self.p_0), (u_g1, self.p_1)]:
                partial1 = np.matmul(u.T, np.linalg.inv(cvm))
                partial1 = np.matmul(partial1, x)
                partial2 = 0.5 * u.T
                partial2 = np.matmul(partial2, np.linalg.inv(cvm))
                partial2 = np.matmul(partial2, u)
                finals.append(float(partial1 - partial2 + np.log(p)))
            if finals[0] > finals[1]:
                ps.append(0)
            else:
                ps.append(1)
        
        return ps

    def case3(self, t):
        # Numpy-friendly mean and covariance formatting
        pts = [np.matrix((x[0], x[1])).T for x in t]
        u_g0 = np.matrix((self.ux_0, self.uy_0)).T
        u_g1 = np.matrix((self.ux_1, self.uy_1)).T
        cvm_0 = self.cv_0
        cvm_1 = self.cv_1
        ps = []

        # Predict each point
        for x in pts:
            finals = []
            for u, p, c in [(u_g0, self.p_0, cvm_0), (u_g1, self.p_1, cvm_1)]:
                partial1 = -0.5 * x.T
                partial1 = np.matmul(partial1, np.linalg.inv(c))
                partial1 = np.matmul(partial1, x)
                partial2 = np.matmul(u.T, np.linalg.inv(c).T)
                partial2 = np.matmul(partial2, x)
                partial3 = 0.5 * u.T
                partial3 = np.matmul(partial3, np.linalg.inv(c))
                partial3 = np.matmul(partial3, u)
                partial4 = 0.5 * np.log(np.linalg.det(c))
                partial4 += np.log(p)
                finals.append(partial1 + partial2 - partial3 - partial4)
            if finals[0] > finals[1]:
                ps.append(0)
            else:
                ps.append(1)
            
        return ps

    # The bimodal case. See report for formula and source
    def case4(self, t):
        pts = [np.matrix((x[0], x[1])).T for x in t]
        u_g00 = np.matrix((self.ux_00, self.uy_00)).T
        u_g01 = np.matrix((self.ux_01, self.uy_01)).T
        u_g10 = np.matrix((self.ux_10, self.uy_10)).T
        u_g11 = np.matrix((self.ux_11, self.uy_11)).T
        us = [(u_g00, u_g01), (u_g10, u_g11)]
        ws = [(self.w_00, self.w_01), (self.w_10, self.w_11)]
        cvms = [(self.cv_00, self.cv_01), (self.cv_10, self.cv_11)]
        prs = [self.p_0, self.p_1]
        params = []
        for i in range(2):
            params.append((us[i], ws[i], cvms[i], prs[i]))
        ps = []

        for x in pts:
            finals = []
            for up, wp, cvp, pp in params:
                partial1_0 = wp[0] / (2 * np.pi * np.sqrt(np.linalg.det(cvp[0])))
                partial2_0 = -0.5 * (x - up[0]).T
                partial2_0 = np.matmul(partial2_0, np.linalg.inv(cvp[0]))
                partial2_0 = np.matmul(partial2_0, (x - up[0]))
                partial_0 = partial1_0 * np.exp(partial2_0)
                partial1_1 = wp[1] / (2 * np.pi * np.sqrt(np.linalg.det(cvp[1])))
                partial2_1 = -0.5 * (x - up[1]).T
                partial2_1 = np.matmul(partial2_1, np.linalg.inv(cvp[1]))
                partial2_1 = np.matmul(partial2_1, (x - up[1]))
                partial_1 = partial1_1 * np.exp(partial2_1)
                final = partial_0 + partial_1 + pp
                finals.append(final)
            if finals[0] > finals[1]:
                ps.append(0)
            else:
                ps.append(1)

        return ps

    # Fit the model
    def fit(self, l):
        c_0 = [x for x in l if x[2] == 0]
        c_1 = [x for x in l if x[2] == 1]
        cl_0 = [(x[0], x[1]) for x in c_0]
        cl_1 = [(x[0], x[1]) for x in c_1]
        xs_0 = [x[0] for x in c_0]
        ys_0 = [x[1] for x in c_0]
        xs_1 = [x[0] for x in c_1]
        ys_1 = [x[1] for x in c_1]
        # Calculate the means
        self.ux_0 = np.mean(xs_0)
        self.uy_0 = np.mean(ys_0)
        self.ux_1 = np.mean(xs_1)
        self.uy_1 = np.mean(ys_1)
        # Calculate the covariance matrices
        self.cv_0 = np.cov(cl_0, rowvar=False)
        self.cv_1 = np.cov(cl_1, rowvar=False)
        # Calculate the scalar variances
        self.v_0 = np.var([(x[0], x[1]) for x in c_0])
        self.v_1 = np.var([(x[0], x[1]) for x in c_1])

    # Fit a bimodal model. Same thing as a regular fit, but more of it.
    def bimodal_fit(self, l):
        c_00 = [x for x in l if x[2] == 0 and x[0] < -0.25]
        c_01 = [x for x in l if x[2] == 0 and x not in c_00]
        c_10 = [x for x in l if x[2] == 1 and x[0] < 0.00]
        c_11 = [x for x in l if x[2] == 1 and x not in c_10]

        cl_00 = [(x[0], x[1]) for x in c_00]
        cl_01 = [(x[0], x[1]) for x in c_01]
        cl_10 = [(x[0], x[1]) for x in c_10]
        cl_11 = [(x[0], x[1]) for x in c_11]

        xs_00 = [x[0] for x in c_00]
        ys_00 = [x[1] for x in c_00]
        xs_01 = [x[0] for x in c_01]
        ys_01 = [x[1] for x in c_01]
        xs_10 = [x[0] for x in c_10]
        ys_10 = [x[1] for x in c_10]
        xs_11 = [x[0] for x in c_11]
        ys_11 = [x[1] for x in c_11]

        self.ux_00 = np.mean(xs_00)
        self.uy_00 = np.mean(ys_00)
        self.ux_01 = np.mean(xs_01)
        self.uy_01 = np.mean(ys_01)
        self.ux_10 = np.mean(xs_10)
        self.uy_10 = np.mean(ys_10)
        self.ux_11 = np.mean(xs_11)
        self.uy_11 = np.mean(ys_11)

        self.cv_00 = np.cov(cl_00, rowvar=False)
        self.cv_01 = np.cov(cl_01, rowvar=False)
        self.cv_10 = np.cov(cl_10, rowvar=False)
        self.cv_11 = np.cov(cl_11, rowvar=False)

        self.v_00 = np.var([(x[0], x[1]) for x in c_00])
        self.v_01 = np.var([(x[0], x[1]) for x in c_01])
        self.v_10 = np.var([(x[0], x[1]) for x in c_10])
        self.v_11 = np.var([(x[0], x[1]) for x in c_11])

        self.w_00 = float(len(c_00)) / (len(c_00) + len(c_01))
        self.w_01 = float(len(c_01)) / (len(c_00) + len(c_01))
        self.w_10 = float(len(c_10)) / (len(c_10) + len(c_11))
        self.w_11 = float(len(c_11)) / (len(c_10) + len(c_11))

    # Dispatch function for predictions based on user input
    def predict(self, t):
        if args.case == 1:
            return self.case1(t)
        elif args.case == 2:
            return self.case2(t)
        elif args.case == 3:
            return self.case3(t)
        elif args.case == 4:
            return self.case4(t)
        else:
            print('Invalid case. Aborting.')

    def __init__(self):
        fit = None
        predict = None
        self.ux_0 = 0
        self.uy_0 = 0
        self.ux_1 = 0
        self.uy_1 = 0
        self.v_0 = 0
        self.v_1 = 0
        self.cv_0 = []
        self.cv_1 = []
        self.p_0 = 0.5
        self.p_1 = 0.5


# Read datasets into a spedcified list
def read_dataset(path, l):
    with open(path, 'r') as df:
        for line in df:
            #print(line)
            if 'xs' in line:
                continue
            sline = line.split()
            l.append((float(sline[0]), float(sline[1]), int(sline[2])))

# Generate initial figures of the testing and unlabeled training data
def generate_early_figs():
    xs_0 = [x[0] for x in training if x[2] == 0]
    ys_0 = [x[1] for x in training if x[2] == 0]
    xs_1 = [x[0] for x in training if x[2] == 1]
    ys_1 = [x[1] for x in training if x[2] == 1]
    plt.scatter(xs_0, ys_0, color='cadetblue', label='Set 0')
    plt.scatter(xs_1, ys_1, color='orange', marker='*', label='Set 1')
    plt.xlabel('x')
    plt.ylabel('y')
    l = plt.legend()
    plt.title('Labeled Training Data')
    plt.savefig('lbl_train_data.pdf')

    plt.cla()
    tex = [x[0] for x in testing]
    tey = [x[1] for x in testing]
    plt.scatter(tex, tey, color='purple', marker='1', label='Test Data')
    plt.xlabel('x')
    plt.ylabel('y')
    l = plt.legend()
    plt.title('Unclassified Test Data')
    plt.savefig('unclassed_test_data.pdf')

# Generate figures on predictions (including decision regions)
def generate_late_figs(l, r, sup='', model=None):
    w_0 = []
    w_1 = []
    for i in range(len(l)):
        if r[i] == 0:
            w_0.append(l[i])
        else:
            w_1.append(l[i])

    xs_0 = [x[0] for x in w_0]
    ys_0 = [x[1] for x in w_0]
    xs_1 = [x[0] for x in w_1]
    ys_1 = [x[1] for x in w_1]
    
    if sup == '':
        additional = ''
    else:
        additional = ' (p(w0) = {})'.format(sup.replace('-', ''))
    plt.cla()
    # Draw the decision boundaries. Big thanks to 
    # https://stackoverflow.com/questions/19054923/plot-decision-boundary-matplotlib
    # for this one.
       
    if model is not None: 
        xmin, xmax = min(xs_0 + xs_1) - 0.1, max(xs_0 + xs_1) + 0.1
        ymin, ymax = min(ys_0 + ys_1) - 0.1, max(ys_0 + ys_1) + 0.1
        xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.1), np.arange(ymin, ymax, 0.1))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = np.array(Z).reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap='inferno', alpha = 0.25, label='Decision boundary')
    plt.scatter(xs_0, ys_0, color='cadetblue', marker='.', label='Set 0 Predicted')
    plt.scatter(xs_1, ys_1, color='orange', marker='*', label='Set 1 Predicted')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Case {} Test Set Predictions{}'.format(args.case, additional))
    l = plt.legend()
    plt.savefig('pred-{}{}-test.pdf'.format(args.case, sup))

    # Draw the decision boundaries. Big thanks to 
    # https://stackoverflow.com/questions/19054923/plot-decision-boundary-matplotlib
    # for this one.
    if model == None:
        plt.savefig('pred-{}{}-test.pdf'.format(args.case, sup))
        return
        
# Plot the ACTUAL test set class membership
def generate_correct_fig():
    xs_0 = [x[0] for x in testing if x[2] == 0]
    ys_0 = [x[1] for x in testing if x[2] == 0]
    xs_1 = [x[0] for x in testing if x[2] == 1]
    ys_1 = [x[1] for x in testing if x[2] == 1]

    plt.gca()
    plt.scatter(xs_0, ys_0, color='cadetblue', marker='.', label='Set 0')
    plt.scatter(xs_1, ys_1, color='orange', marker='*', label='Set 1')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Actual Test Set Class Membership')
    l = plt.legend()
    plt.savefig('actual_test_membership.pdf')

# Do a prior probability sweet from 0.05 to 0.95. Generate strategic figures
# and save them, and print a table showing model accuracy
def do_prior_sweep(M):
    ps = np.arange(0.05, 1.0, 0.05)
    start = time.time()
    for p in ps:
        M.p_0 = p
        M.p_1 = 1 - p
        if args.case > 0 and args.case < 4:
            M.fit(training)
        else:
            M.bimodal_fit(training)
        predictions = M.predict(testing)
        right = 0
        for i in range(len(predictions)):
            if predictions[i] == testing[i][2]:
                right += 1
        print('{:5.2}{:5.2}{:10.7}'.format(p, 1 - p, float(right) / len(predictions)))
        if np.isclose(p, 0.05) or np.isclose(p, 0.25) or np.isclose(p, 0.50) or np.isclose(p, 0.75) or np.isclose(p, 0.95):
            generate_late_figs(testing, predictions, sup='-{}'.format(round(p, 2)), model=M)
            pass
    end = time.time()
    print('{} iters in {} seconds'.format(len(ps), end - start))

# Get a slate of statistical calues for a dataset
def get_stat_values(l):
    cl = [(x[0], x[1]) for x in l]
    xs = [x[0] for x in l]
    ys = [x[1] for x in l]
    ux = np.mean(xs)
    uy = np.mean(ys)
    cv = np.cov(cl, rowvar=False)
    v = np.var([(x[0], x[1]) for x in l])
    return xs, ys, ux, uy, v, cv

def main():
    # Read files
    read_dataset(args.train, training)
    read_dataset(args.test, testing)

    # Generate basic figures
    generate_early_figs()

    # Train the model
    M = Model()
    if args.case > 0 and args.case < 4:
        M.fit(training)
        print(M.v_0)
        print(M.v_1)
        print(M.cv_0)
        print(M.cv_1)
    elif args.case == 4:
        M.bimodal_fit(training)
        print(M.cv_00)
        print(M.cv_01)
        print(M.cv_10)
        print(M.cv_11)
    else:
        print('Invalid case value {}. Must be 1-4.'.format(args.case))

    # Predict and calculate accuracy
    predictions = M.predict(testing)
    right = 0
    for i in range(len(predictions)):
        if predictions[i] == testing[i][2]:
            right += 1

    # Generate the final figures
    generate_late_figs(testing, predictions, model=M)
    generate_correct_fig()

    print('Accuracy: {}'.format(float(right) / len(predictions)))
    do_prior_sweep(M)
    return 0
