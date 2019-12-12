import csv
import numpy as np
import timeit
import random
import itertools
import math
from random import randint
from dim_reduction import *
from knn import KNN
from mpp import MPP
from sklearn import svm
from sklearn.metrics import roc_curve
from sklearn import tree
from bpnn import Network
from kmeans import KMeans
from kohonen import KMap
from wta import WTA
import matplotlib as mpl
import matplotlib.pyplot as plt

#mpl.use('Qt4Agg')

def filter_retweets(data):
    no_rt = []
    for sample in data:
        retweet = sample[2]
        if retweet == 'False':
            no_rt.append(sample)
    return no_rt

def extract_features(data):
    features = np.zeros((9,len(data)))
    for i in range(0,len(data)):
        tweet = data[i][3]
        upper = 0
        for word in tweet.split():
            if word.isupper():
                upper += 1
        features[0,i] = tweet.count('!')
        features[1,i] = tweet.lower().count('pic.twitter.com')
        features[2,i] = tweet.count('@')
        features[3,i] = upper
        features[4,i] = tweet.lower().count('http')
        features[5,i] = tweet.count('#')
        features[6,i] = tweet.count('"')
        features[7,i] = tweet.count(',')
        features[8,i] = tweet.count('.')
#        features[7,i] = tweet.lower().count('trump') + tweet.lower().count('donald') 
#        features[8,i] = tweet.lower().count('maga') + tweet.lower().count('make america great again') + tweet.lower().count('makeamericagreatagain') + tweet.lower().count('make #americagreatagain') + tweet.lower().count('make america') + tweet.lower().count('great again')
 #       features[8,i] = tweet.lower().count('loser')
    return features

def nb_fusion(conf_mat, labels):
    num_classifiers = conf_mat.shape[0]
    comb = []
    for i in range(0,num_classifiers):
        comb.append(list(range(2)))
    comb = list(itertools.product(*comb))
    table = np.zeros((2,len(comb)))
    num_samples = len(labels)
    num1 = np.count_nonzero(labels)
    num0 = num_samples - num1
    for i in range(0,len(comb)):
        prob0 = (1/math.pow(num0,num_classifiers-1))
        prob1 = (1/math.pow(num1,num_classifiers-1))
        prod = np.ones((2,1))
        for j in range(0,num_classifiers):
            col = comb[i][j]
            print(prod.shape)
            print(conf_mat[j,:,col].reshape((2,1)).shape)
            print(np.multiply(prod, conf_mat[j,:,col].reshape((2,1))).shape)
            prod = np.multiply(prod, conf_mat[j,:,col].reshape((2,1)))
        prod[0] = prod[0] * prob0
        prod[1] = prod[1] * prob1
        table[:,i] = prod[:,0]
    return table,comb

def majority_vote(predictions):
    num_classifiers = predictions.shape[0]
    num_samples = predictions.shape[1]
    fused = np.zeros(num_samples)
    for i in range(0, num_samples):
        yes = 0
        no = 0
        for j in range(0, num_classifiers):
            if predictions[j,i] == 0.0:
                no += 1
            else:
                yes += 1
        if yes > no:
            fused[i] = 1.0
        else:
            fused[i] = 0.0
    return fused

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

def confusion_matrix(predict, true):
    tp,tn,fn,fp = perf_eval(predict, true)
    conf_mat = np.zeros((2,2))
    conf_mat[0,0] = tp
    conf_mat[0,1] = fp
    conf_mat[1,0] = fn
    conf_mat[1,1] = tn
    return conf_mat

def m_fold_cross_validation(tweets, person, m):
    print(len(tweets[0]))
    print(len(tweets[1]))
    print(len(tweets[2]))
    print(len(tweets[3]))
    print(len(tweets[4]))
    print(len(tweets[5]))
    all_tweets = []
    all_tweets.extend(tweets[0])
    all_tweets.extend(tweets[1])
    all_tweets.extend(tweets[2])
    all_tweets.extend(tweets[3])
    all_tweets.extend(tweets[4])
    all_tweets.extend(tweets[5])
    y = [0]*len(all_tweets)
    start = 0
    end = 0
    for i in range(0,person):
        start += len(tweets[i])
    end = start + len(tweets[person])
    print(start)
    print(end)
    for i in range(start, end):
        y[i] = 1.0
    z = list(zip(all_tweets, y))
    random.shuffle(z)
    all_tweets, all_labels = zip(*z)
    num_per_set = int(len(all_tweets)/m)
    all_tweets = all_tweets[0:num_per_set*m]
    all_labels = all_labels[0:num_per_set*m]
    sets = []
    for i in range(0,m):
        start = i*num_per_set
        end = (i+1)*num_per_set
        train_tweets = all_tweets[0:start] + all_tweets[end:]
        train_labels = all_labels[0:start] + all_labels[end:]
        test_tweets = all_tweets[start:end]
        test_labels = all_labels[start:end]
        train = (train_tweets, train_labels)
        test = (test_tweets, test_labels)
        sets.append((train, test))
    return sets

def create_dataset(tweets, person, num_train_tweets, train_percentages, num_test_tweets, test_percentages):
    random.shuffle(tweets[0])
    random.shuffle(tweets[1])
    random.shuffle(tweets[2])
    random.shuffle(tweets[3])
    random.shuffle(tweets[4])
    random.shuffle(tweets[5])

    train_data = []
    test_data = []

    num_train_0 = int(train_percentages[0]*num_train_tweets)
    num_train_1 = int(train_percentages[1]*num_train_tweets)
    num_train_2 = int(train_percentages[2]*num_train_tweets)
    num_train_3 = int(train_percentages[3]*num_train_tweets)
    num_train_4 = int(train_percentages[4]*num_train_tweets)
    num_train_5 = int(train_percentages[5]*num_train_tweets)

    num_test_0 = int(test_percentages[0]*num_test_tweets)
    num_test_1 = int(test_percentages[1]*num_test_tweets)
    num_test_2 = int(test_percentages[2]*num_test_tweets)
    num_test_3 = int(test_percentages[3]*num_test_tweets)
    num_test_4 = int(test_percentages[4]*num_test_tweets)
    num_test_5 = int(test_percentages[5]*num_test_tweets)

    for i in range(0, num_train_0):
        train_data.append(tweets[0][i])
    for i in range(num_train_0, num_train_0+num_test_0):
        test_data.append(tweets[0][i])

    for i in range(0, num_train_1):
        train_data.append(tweets[1][i])
    for i in range(num_train_1, num_train_1+num_test_1):
        test_data.append(tweets[1][i])

    for i in range(0, num_train_2):
        train_data.append(tweets[2][i])
    for i in range(num_train_2, num_train_2+num_test_2):
        test_data.append(tweets[2][i])

    for i in range(0, num_train_3):
        train_data.append(tweets[3][i])
    for i in range(num_train_3, num_train_3+num_test_3):
        test_data.append(tweets[3][i])

    for i in range(0, num_train_4):
        train_data.append(tweets[4][i])
    for i in range(num_train_4, num_train_4+num_test_4):
        test_data.append(tweets[4][i])

    for i in range(0, num_train_5):
        train_data.append(tweets[5][i])
    for i in range(num_train_5, num_train_5+num_test_5):
        test_data.append(tweets[5][i])
    
    train_labels = np.zeros(len(train_data))
    start = int(np.sum(train_percentages[0:person])*num_train_tweets)
    end = int(np.sum(train_percentages[0:person+1])*num_train_tweets)
    for i in range(start, end):
        train_labels[i] = 1
    
    test_labels = np.zeros(len(test_data))
    start = int(np.sum(test_percentages[0:person])*num_test_tweets)
    end = int(np.sum(test_percentages[0:person+1])*num_test_tweets)
    for i in range(start, end):
        test_labels[i] = 1

    return [(train_data, train_labels), (test_data, test_labels)]

def plot_roc(f_rate, t_rate):
    plt.plot(f_rate, t_rate, color='orange', label='ROC')
    plt.plot([0,1],[0,1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

def main():
    dt_tweets = []
    hc_tweets = []
    kk_tweets = []
    ndgt_tweets = []
    rd_tweets = []
    sk_tweets = []
    with open('DonaldTrumpDataSet.csv', 'r', encoding='utf8', errors='ignore') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            dt_tweets.append(row)
    with open('HillaryClintonDataSet.csv', 'r', encoding='utf8', errors='ignore') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            hc_tweets.append(row)
    with open('KimKardashianDataSet.csv', 'r', encoding='utf8', errors='ignore') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            kk_tweets.append(row)
    with open('NeildeGrasseTysonDataSet.csv', 'r', encoding='utf8', errors='ignore') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            ndgt_tweets.append(row)
    with open('RichardDawkinsDataSet.csv', 'r', encoding='utf8', errors='ignore') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            rd_tweets.append(row)
    with open('ScottKellyDataSet.csv', 'r', encoding='utf8', errors='ignore') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            sk_tweets.append(row)

    dt_tweets.pop(0)
    hc_tweets.pop(0)
    kk_tweets.pop(0)
    ndgt_tweets.pop(0)
    rd_tweets.pop(0)
    sk_tweets.pop(0)

#    print(len(dt_tweets))
#    print(len(hc_tweets))
#    print(len(kk_tweets))
#    print(len(ndgt_tweets))
#    print(len(rd_tweets))
#    print(len(sk_tweets))
#    print(len(dt_tweets) + len(hc_tweets) + len(kk_tweets) + len(ndgt_tweets) + len(rd_tweets) + len(sk_tweets))
    tweets = [dt_tweets, hc_tweets, kk_tweets, ndgt_tweets, rd_tweets, sk_tweets]

    dt_nort_tweets   = filter_retweets(dt_tweets)
    hc_nort_tweets   = filter_retweets(hc_tweets)
    kk_nort_tweets   = filter_retweets(kk_tweets)
    ndgt_nort_tweets = filter_retweets(ndgt_tweets)
    rd_nort_tweets   = filter_retweets(rd_tweets)
    sk_nort_tweets   = filter_retweets(sk_tweets)

#    print(len(dt_nort_tweets) + len(hc_nort_tweets) + len(kk_nort_tweets) + len(ndgt_nort_tweets) + len(rd_nort_tweets) + len(sk_nort_tweets))
    nort_tweets = [dt_nort_tweets, hc_nort_tweets, kk_nort_tweets, ndgt_nort_tweets, rd_nort_tweets, sk_nort_tweets]

#    percentages = [0.43, 0.08, 0.26, 0.06, 0.14, 0.03]
    percentages = [0.17, 0.17, 0.17, 0.17, 0.16, 0.16]
    datasets = create_dataset(tweets, 0, 7000, percentages, 500, percentages)
    nort_datasets = create_dataset(nort_tweets, 0, 7000, percentages, 500, percentages)
    train_set = datasets[0][0]
    train_labels = datasets[0][1]
    test_set = datasets[1][0]
    test_labels = datasets[1][1]

    nort_train_set = datasets[0][0]
    nort_train_labels = datasets[0][1]
    nort_test_set = datasets[1][0]
    nort_test_labels = datasets[1][1]

    data = train_set
    true_labels = train_labels
    test_data = test_set
    test_labels = test_labels

    nort_data = nort_train_set
    nort_true_labels = nort_train_labels
    nort_test_data = nort_test_set
    nort_test_labels = nort_test_labels
    
    features = extract_features(data)
    nort_features = extract_features(nort_data)
    test_features = extract_features(test_data)
    test_features2 = test_features

    mean = np.mean(features, axis=1).reshape((features.shape[0],1))
    sigma = np.std(features, axis=1).reshape((features.shape[0],1))
    mean2 = np.mean(nort_features, axis=1).reshape((nort_features.shape[0],1))
    sigma2 = np.std(nort_features, axis=1).reshape((nort_features.shape[0],1))
    standardize(features, mean, sigma)
    standardize(nort_features, mean2, sigma2)
    standardize(test_features, mean, sigma)
    standardize(test_features2, mean2, sigma2)

#    fld = FLD()
#    fld.setup(features, true_labels)
#    features = fld.reduce(features)
#    test_features = fld.reduce(test_features)
#
#    fld2 = FLD()
#    fld2.setup(nort_features, nort_train_labels)
#    nort_features = fld.reduce(nort_features)
#    test_features2 = fld.reduce(test_features2)

#    pca = PCA()
#    pca.setup(features, 0.8)
#    features = pca.reduce(features)
#    test_features = pca.reduce(test_features)
#    print(pca.eigenvalues)
#
#    pca2 = PCA()
#    pca2.setup(nort_features, 0.8)
#    nort_features = pca.reduce(nort_features)
#    test_features2 = pca.reduce(test_features2)
#    print(pca2.eigenvalues)

#    print("Decision Tree")
#    clf = tree.DecisionTreeClassifier()
#    clf.probability = True
#    clf.fit(features.T, true_labels)
#    ymodel = clf.predict(test_features.T)
#    prob = clf.predict_proba(test_features.T)
#    fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#    plt.figure()
#    plot_roc(fper, tper)
#    tp,tn,fn,fp = perf_eval(ymodel, test_labels)
#    print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#    print('TP:',tp)
#    print('TN:',tn)
#    print('FP:',fp)
#    print('FN:',fn)

#    print("SVM linear")
#    clf = svm.SVC(kernel='linear', gamma='auto')
#    clf.probability = True
#    clf.fit(features.T, true_labels)
#    ymodel = clf.predict(test_features.T)
#    prob = clf.predict_proba(test_features.T)
#    fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#    plt.figure()
#    plot_roc(fper, tper)
#    tp,tn,fn,fp = perf_eval(ymodel, test_labels)
#    print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#    print('TP:',tp)
#    print('TN:',tn)
#    print('FP:',fp)
#    print('FN:',fn)

#    print("SVM poly")
#    clf = svm.SVC(kernel='poly', gamma='auto')
#    clf.probability = True
#    clf.fit(features.T, true_labels)
#    ymodel = clf.predict(test_features.T)
#    prob = clf.predict_proba(test_features.T)
#    fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#    plt.figure()
#    plot_roc(fper, tper)
#    tp,tn,fn,fp = perf_eval(ymodel, test_labels)
#    print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#    print('TP:',tp)
#    print('TN:',tn)
#    print('FP:',fp)
#    print('FN:',fn)
#
#    print("SVM rbf")
#    clf = svm.SVC(kernel='rbf', gamma='auto')
#    clf.probability = True
#    clf.fit(features.T, true_labels)
#    ymodel = clf.predict(test_features.T)
#    prob = clf.predict_proba(test_features.T)
#    fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#    plt.figure()
#    plot_roc(fper, tper)
#    tp,tn,fn,fp = perf_eval(ymodel, test_labels)
#    print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#    print('TP:',tp)
#    print('TN:',tn)
#    print('FP:',fp)
#    print('FN:',fn)
#
#    print("SVM sigmoid")
#    clf = svm.SVC(kernel='sigmoid', gamma='auto')
#    clf.probability = True
#    clf.fit(features.T, true_labels)
#    ymodel = clf.predict(test_features.T)
#    prob = clf.predict_proba(test_features.T)
#    fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#    plt.figure()
#    plot_roc(fper, tper)
#    tp,tn,fn,fp = perf_eval(ymodel, test_labels)
#    print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#    print('TP:',tp)
#    print('TN:',tn)
#    print('FP:',fp)
#    print('FN:',fn)

#    k = 3
#    print("KNN: k =",k)
#    print('2 norm')
#    knn_model = KNN(k)
#    knn_model.fit(features, true_labels)
#    ymodel = knn_model.predict(test_features, norm=2)
#    prob = knn_model.predict_prob(test_features)
#    print(prob)
#    fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#    plt.figure()
#    plot_roc(fper, tper)
#    tp,tn,fn,fp = perf_eval(ymodel, test_labels)
#    print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#    print('TP:',tp)
#    print('TN:',tn)
#    print('FP:',fp)
#    print('FN:',fn)

#    knn_model2 = KNN(k)
#    knn_model2.fit(nort_features, nort_train_labels)
#    ymodel = knn_model2.predict(test_features2, norm=2)
#    prob = knn_model2.predict_prob(test_features)
#    print(prob)
#    fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#    plt.figure()
#    plot_roc(fper, tper)
#    tp,tn,fn,fp = perf_eval(ymodel, test_labels)
#    print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#    print('TP:',tp)
#    print('TN:',tn)
#    print('FP:',fp)
#    print('FN:',fn)

#    print('inf norm')
#    knn_model = KNN(k)
#    knn_model.fit(features, true_labels)
#    ymodel = knn_model.predict(test_features, norm='inf')
#    prob = knn_model.predict_prob(test_features)
#    print(prob)
#    fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#    plt.figure()
#    plot_roc(fper, tper)
#    tp,tn,fn,fp = perf_eval(ymodel, test_labels)
#    print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#    print('TP:',tp)
#    print('TN:',tn)
#    print('FP:',fp)
#    print('FN:',fn)
#
#    knn_model2 = KNN(k)
#    knn_model2.fit(nort_features, nort_train_labels)
#    ymodel = knn_model2.predict(test_features2, norm='inf')
#    prob = knn_model2.predict_prob(test_features)
#    print(prob)
#    fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#    plt.figure()
#    plot_roc(fper, tper)
#    tp,tn,fn,fp = perf_eval(ymodel, test_labels)
#    print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#    print('TP:',tp)
#    print('TN:',tn)
#    print('FP:',fp)
#    print('FN:',fn)
#
#    print('1 norm')
#    knn_model = KNN(k)
#    knn_model.fit(features, true_labels)
#    ymodel = knn_model.predict(test_features, norm=1)
#    prob = knn_model.predict_prob(test_features)
#    print(prob)
#    fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#    plt.figure()
#    plot_roc(fper, tper)
#    tp,tn,fn,fp = perf_eval(ymodel, test_labels)
#    print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#    print('TP:',tp)
#    print('TN:',tn)
#    print('FP:',fp)
#    print('FN:',fn)
#
#    knn_model2 = KNN(k)
#    knn_model2.fit(nort_features, nort_train_labels)
#    ymodel = knn_model2.predict(test_features2, norm=1)
#    prob = knn_model.predict_prob(test_features)
#    print(prob)
#    fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#    plt.figure()
#    plot_roc(fper, tper)
#    tp,tn,fn,fp = perf_eval(ymodel, test_labels)
#    print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#    print('TP:',tp)
#    print('TN:',tn)
#    print('FP:',fp)
#    print('FN:',fn)

#    true = np.count_nonzero(true_labels)/true_labels.shape[0]
#    false = 1-true
#    print("MPP case 1")
#    mpp = MPP(1)
#    mpp.set_prior(false, true)
#    mpp.fit(features, true_labels)
#    mpp_pred1 = mpp.predict(test_features)
#    prob = mpp.predict_prob(test_features)
#    fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#    plt.figure()
#    plot_roc(fper, tper)
#    tp,tn,fn,fp = perf_eval(mpp_pred1, test_labels)
#    print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#    print('TP:',tp)
#    print('TN:',tn)
#    print('FP:',fp)
#    print('FN:',fn)
#
#    print("MPP case 2")
#    mpp = MPP(2)
#    mpp.set_prior(false, true)
#    mpp.fit(features, true_labels)
#    mpp_pred2 = mpp.predict(test_features)
#    prob = mpp.predict_prob(test_features)
#    fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#    plt.figure()
#    plot_roc(fper, tper)
#    tp,tn,fn,fp = perf_eval(mpp_pred2, test_labels)
#    print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#    print('TP:',tp)
#    print('TN:',tn)
#    print('FP:',fp)
#    print('FN:',fn)
#
#    print("MPP case 3")
#    mpp = MPP(3)
#    mpp.set_prior(false, true)
#    mpp.fit(features, true_labels)
#    mpp_pred3 = mpp.predict(test_features)
#    prob = mpp.predict_prob(test_features)
#    fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#    plt.figure()
#    plot_roc(fper, tper)
#    tp,tn,fn,fp = perf_eval(mpp_pred3, test_labels)
#    print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#    print('TP:',tp)
#    print('TN:',tn)
#    print('FP:',fp)
#    print('FN:',fn)
#
#    print("Fused MPP")
#    mpp_predictions = np.zeros((3,mpp_pred1.shape[0]))
#    mpp_predictions[0,:] = mpp_pred1.T
#    mpp_predictions[1,:] = mpp_pred2.T
#    mpp_predictions[2,:] = mpp_pred3.T
#    mpp_fused = majority_vote(mpp_predictions)
#    tp,tn,fn,fp = perf_eval(mpp_fused, test_labels)
#    print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#    print('TP:',tp)
#    print('TN:',tn)
#    print('FP:',fp)
#    print('FN:',fn)


#    print("BGNN")
#    num_features = 7
#    net = Network([features.shape[0], 10, 2])
#    net.SGD(features, true_labels, 1000, 1, 0.05, test_features, test_labels)
#    prob = net.SGD_prob(features, true_labels, 100, 1, 0.10, test_features, test_labels)
#    fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#    plt.figure()
#    plot_roc(fper, tper)

    #plt.show()
    kmeans = KMeans(2)
    kmeans.predict(features, true_labels)
    kmeans.predict(test_features, test_labels)
#    kmap.predict(test_features, test_labels, e=0.0000001, iters=1000)
#    wta = WTA(2)
#    wta.predict(test_features, test_labels, e=0.01)
#    kmap = KMap(2)
#    kmap.predict(test_features, test_labels, e=0.001, iters=100)
#    kmap.predict(test_features, test_labels, e=0.0000001, iters=1000)

#    m = 5
#    sets = m_fold_cross_validation(tweets, 0, m)
#    print(len(sets))
#    conf_mats = np.zeros((m,2,2))
#    for i in range(0,m):
#        train,test = sets[i]
#        train_tweets,train_labels = train
#        test_tweets,test_labels = test
#        train_features = extract_features(train_tweets)
#        test_features = extract_features(test_tweets)
#        mean = np.mean(train_features, axis=1).reshape((train_features.shape[0],1))
#        sigma = np.std(train_features, axis=1).reshape((train_features.shape[0],1))
#        standardize(train_features, mean, sigma)
#        standardize(test_features, mean, sigma)
#        print("BGNN")
#        net = Network([train_features.shape[0], 10, 2])
#        conf_mats[i,:,:] = net.SGD(train_features, train_labels, 1000, 1, 0.05, test_features, test_labels)
#    kmap.predict(test_features, test_labels, e=0.0000001, iters=1000)

    m = 10
    sets = m_fold_cross_validation(tweets, 0, m)
    print(len(sets))
    num_test = len(sets[0][0][1])
    for i in range(0,1):
        train,test = sets[i]
        train_tweets,train_labels = train
        test_tweets,test_labels = test
        train_features = extract_features(train_tweets)
        test_features = extract_features(test_tweets)
        mean = np.mean(train_features, axis=1).reshape((train_features.shape[0],1))
        sigma = np.std(train_features, axis=1).reshape((train_features.shape[0],1))
        standardize(train_features, mean, sigma)
        standardize(test_features, mean, sigma)

#        fld = FLD()
#        fld.setup(train_features, train_labels)
#        train_features = fld.reduce(train_features)
#        test_features = fld.reduce(test_features)
    
#        pca = PCA()
#        pca.setup(train_features, 0.8)
#        train_features = pca.reduce(train_features)
#        test_features = pca.reduce(test_features)
#        print(pca.eigenvalues)
   
#        conf_mats = np.zeros((2,2,2))
#        all_labels = np.zeros((2,num_test))

#        print("Decision Tree")
#        clf = tree.DecisionTreeClassifier()
#        clf.probability = True
#        clf.fit(train_features.T, train_labels)
#        ymodel = clf.predict(test_features.T)
#        prob = clf.predict_proba(test_features.T)
#        fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#        plt.figure()
#        plot_roc(fper, tper)
#        tp,tn,fn,fp = perf_eval(ymodel, test_labels)
#        print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#        print('TP:',tp)
#        print('TN:',tn)
#        print('FP:',fp)
#        print('FN:',fn)

        kmeans = KMeans(2)
        kmeans.predict(test_features, test_labels)
        wta = WTA(2)
        wta.predict(test_features, test_labels, e=0.01)
        kmap = KMap(2)
        kmap.predict(test_features, test_labels, e=0.001, iters=100)
#
#        predictions = np.zeros((2,ymodel.shape[0]))
#        predictions[0,:] = ymodel.T
#        fused = majority_vote(predictions)
#        tp,tn,fn,fp = perf_eval(fused, test_labels)
#        print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#        print('TP:',tp)
#        print('TN:',tn)
#        print('FP:',fp)
#        print('FN:',fn)

#        print("SVM linear")
#        clf = svm.SVC(kernel='linear', gamma='auto')
#        clf.probability = True
#        clf.fit(train_features.T, train_labels)
#        ymodel = clf.predict(test_features.T)
#        prob = clf.predict_proba(test_features.T)
#        fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#        plt.figure()
#        plot_roc(fper, tper)
#        tp,tn,fn,fp = perf_eval(ymodel, test_labels)
#        print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#        print('TP:',tp)
#        print('TN:',tn)
#        print('FP:',fp)
#        print('FN:',fn)
#   
#        print("SVM poly")
#        clf = svm.SVC(kernel='poly', gamma='auto')
#        clf.probability = True
#        clf.fit(train_features.T, train_labels)
#        ymodel = clf.predict(test_features.T)
#        prob = clf.predict_proba(test_features.T)
#        fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#        plt.figure()
#        plot_roc(fper, tper)
#        tp,tn,fn,fp = perf_eval(ymodel, test_labels)
#        print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#        print('TP:',tp)
#        print('TN:',tn)
#        print('FP:',fp)
#        print('FN:',fn)
#   
#        print("SVM rbf")
#        clf = svm.SVC(kernel='rbf', gamma='auto')
#        clf.probability = True
#        clf.fit(train_features.T, train_labels)
#        ymodel = clf.predict(test_features.T)
#        prob = clf.predict_proba(test_features.T)
#        fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#        plt.figure()
#        plot_roc(fper, tper)
#        tp,tn,fn,fp = perf_eval(ymodel, test_labels)
#        print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#        print('TP:',tp)
#        print('TN:',tn)
#        print('FP:',fp)
#        print('FN:',fn)
#   
#        print("SVM sigmoid")
#        clf = svm.SVC(kernel='sigmoid', gamma='auto')
#        clf.probability = True
#        clf.fit(train_features.T, train_labels)
#        ymodel = clf.predict(test_features.T)
#        prob = clf.predict_proba(test_features.T)
#        fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#        plt.figure()
#        plot_roc(fper, tper)
#        tp,tn,fn,fp = perf_eval(ymodel, test_labels)
#        print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#        print('TP:',tp)
#        print('TN:',tn)
#        print('FP:',fp)
#        print('FN:',fn)

#        k = 3
#        print("KNN: k =",k)
#        print('2 norm')
#        knn_model = KNN(k)
#        knn_model.fit(features, true_labels)
#        ymodel = knn_model.predict(test_features, norm=2)
#        prob = knn_model.predict_prob(test_features)
#        fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#        plt.figure()
#        plot_roc(fper, tper)
#        tp,tn,fn,fp = perf_eval(ymodel, test_labels)
#        print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#        print('TP:',tp)
#        print('TN:',tn)
#        print('FP:',fp)
#        print('FN:',fn)

        print("BPNN")
        net = Network([train_features.shape[0], 10, 10, 2])
        conf_mats[0,:,:],bpnn_pred = net.SGD(train_features, train_labels, 1000, 1, 0.05, test_features, test_labels)
#        all_labels[0,:] = np.array(bpnn_pred)
#        prob = net.SGD_prob(train_features, train_labels, 100, 1, 0.10, test_features, test_labels)
#        fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#        plt.figure()
#        plot_roc(fper, tper)

#        train_labels = np.array(train_labels)
#        test_labels = np.array(test_labels)
#        true = np.count_nonzero(train_labels)/train_labels.shape[0]
#        false = 1-true
#        print("MPP case 1")
#        mpp = MPP(1)
#        mpp.set_prior(false, true)
#        mpp.fit(train_features, train_labels)
#        mpp_pred1 = mpp.predict(test_features)
#        prob = mpp.predict_prob(test_features)
#        fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#        tp,tn,fn,fp = perf_eval(mpp_pred1, test_labels)
#        print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#        print('TP:',tp)
#        print('TN:',tn)
#        print('FP:',fp)
#        print('FN:',fn)
#        plt.figure()
#        plot_roc(fper, tper)
#        print("MPP case 2")
#        mpp = MPP(2)
#        mpp.set_prior(false, true)
#        mpp.fit(train_features, train_labels)
#        mpp_pred2 = mpp.predict(test_features)
#        prob = mpp.predict_prob(test_features)
#        fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#        tp,tn,fn,fp = perf_eval(mpp_pred2, test_labels)
#        print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#        print('TP:',tp)
#        print('TN:',tn)
#        print('FP:',fp)
#        print('FN:',fn)
#        plt.figure()
#        plot_roc(fper, tper)
#        print("MPP case 3")
#        mpp = MPP(3)
#        mpp.set_prior(false, true)
#        mpp.fit(train_features, train_labels)
#        mpp_pred3 = mpp.predict(test_features)
#        prob = mpp.predict_prob(test_features)
#        fper, tper, thresh = roc_curve(test_labels, prob[:,1], pos_label=1)
#        tp,tn,fn,fp = perf_eval(mpp_pred3, test_labels)
#        print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))
#        print('TP:',tp)
#        print('TN:',tn)
#        print('FP:',fp)
#        print('FN:',fn)
#        plt.figure()
#        plot_roc(fper, tper)
#    print(conf_mats)
    print(conf_mats)
    plt.show()


if __name__ == "__main__":
    main()
