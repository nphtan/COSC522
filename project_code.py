import csv
import numpy as np
import timeit
from random import randint
from dim_reduction import *
from knn import KNN
from mpp import MPP

def filter_retweets(data):
    no_rt = []
    for sample in data:
        retweet = sample[2]
        if retweet == 'False':
            no_rt.append(sample)
        else:
            print("retweet")
    return no_rt

def extract_features(data):
    features = np.zeros((7,len(data)))
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
        features[6,i] = len(tweet) 
#        features[6,i] = tweet.lower().count('maga') + tweet.lower().count('make america great again') + tweet.lower().count('makeamericagreatagain') + tweet.lower().count('make #americagreatagain') + tweet.lower().count('make america') + tweet.lower().count('great again')
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
    print(len(dt_tweets))
    print(len(hc_tweets))
    print(len(kk_tweets))
    print(len(ndgt_tweets))
    print(len(rd_tweets))
    print(len(sk_tweets))

    dt_nort_tweets   = filter_retweets(dt_tweets)
    hc_nort_tweets   = filter_retweets(hc_tweets)
    kk_nort_tweets   = filter_retweets(kk_tweets)
    ndgt_nort_tweets = filter_retweets(ndgt_tweets)
    rd_nort_tweets   = filter_retweets(rd_tweets)
    sk_nort_tweets   = filter_retweets(sk_tweets)

    num_train_tweets = 12000
    num_train_person = 2000
    num_train_other = int((num_train_tweets - num_train_person)/5)

    num_test_tweets = 1200
    num_test_person = 200
    num_test_other = int((num_test_tweets - num_test_person)/5)

    data = []
    for i in range(0,num_train_other):
        data.append(kk_tweets[randint(0,len(kk_tweets)-1)])

    for i in range(0,num_train_person):
        data.append(dt_tweets[randint(0,len(dt_tweets)-1)])

    for i in range(0,num_train_other):
        data.append(hc_tweets[randint(0,len(hc_tweets)-1)])

    for i in range(0,num_train_other):
        data.append(ndgt_tweets[randint(0,len(ndgt_tweets)-1)])

    for i in range(0,num_train_other):
        data.append(rd_tweets[randint(0,len(rd_tweets)-1)])

    for i in range(0,num_train_other):
        data.append(sk_tweets[randint(0,len(sk_tweets)-1)])
    
    true_labels = np.zeros(len(data))
    for i in range(0, num_train_person):
        true_labels[i] = 1

    nort_data = []
    for i in range(0,num_train_other):
        nort_data.append(kk_nort_tweets[randint(0,len(kk_nort_tweets)-1)])

    for i in range(0,num_train_person):
        nort_data.append(dt_nort_tweets[randint(0,len(dt_nort_tweets)-1)])

    for i in range(0,num_train_other):
        nort_data.append(hc_nort_tweets[randint(0,len(hc_nort_tweets)-1)])

    for i in range(0,num_train_other):
        nort_data.append(ndgt_nort_tweets[randint(0,len(ndgt_nort_tweets)-1)])

    for i in range(0,num_train_other):
        nort_data.append(rd_nort_tweets[randint(0,len(rd_nort_tweets)-1)])

    for i in range(0,num_train_other):
        nort_data.append(sk_nort_tweets[randint(0,len(sk_nort_tweets)-1)])
    
    nort_train_labels = np.zeros(len(nort_data))
    for i in range(0, num_train_person):
        nort_train_labels[i] = 1

    test_data = []
    for i in range(0,num_test_other):
        test_data.append(kk_tweets[randint(0,len(kk_tweets)-1)])

    for i in range(0,num_test_person):
        test_data.append(dt_tweets[randint(0,len(dt_tweets)-1)])

    for i in range(0,num_test_other):
        test_data.append(hc_tweets[randint(0,len(hc_tweets)-1)])

    for i in range(0,num_test_other):
        test_data.append(ndgt_tweets[randint(0,len(ndgt_tweets)-1)])

    for i in range(0,num_test_other):
        test_data.append(rd_tweets[randint(0,len(rd_tweets)-1)])

    for i in range(0,num_test_other):
        test_data.append(sk_tweets[randint(0,len(sk_tweets)-1)])
    
    test_labels = np.zeros(len(test_data))
    for i in range(0, num_test_person):
        test_labels[i] = 1

    nort_test_data = []
    for i in range(0,num_test_other):
        nort_test_data.append(kk_nort_tweets[randint(0,len(kk_nort_tweets)-1)])

    for i in range(0,num_test_person):
        nort_test_data.append(dt_nort_tweets[randint(0,len(dt_nort_tweets)-1)])

    for i in range(0,num_test_other):
        nort_test_data.append(hc_nort_tweets[randint(0,len(hc_nort_tweets)-1)])

    for i in range(0,num_test_other):
        nort_test_data.append(ndgt_nort_tweets[randint(0,len(ndgt_nort_tweets)-1)])

    for i in range(0,num_test_other):
        nort_test_data.append(rd_nort_tweets[randint(0,len(rd_nort_tweets)-1)])

    for i in range(0,num_test_other):
        nort_test_data.append(sk_nort_tweets[randint(0,len(sk_nort_tweets)-1)])
    
    nort_test_labels = np.zeros(len(nort_test_data))
    for i in range(0, num_test_person):
        nort_test_labels[i] = 1

#    data = []
#    test_data = []
#    with open('ObTr1.csv', 'r', encoding='utf8', errors='ignore') as csvfile:
#        reader = csv.reader(csvfile, delimiter=',')
#        for row in reader:
#            data.append(row)
#    with open('trump_hillary_tweets.csv', 'r', encoding='utf8', errors='ignore') as csvfile:
#        reader = csv.reader(csvfile, delimiter=',')
#        for row in reader:
#            test_data.append(row)
#    
#    data.pop(0)
#    test_data.pop(0)
#
#    full_data = data
#    no_rt_data = filter_retweets(data)
#
#    test_labels = np.ones(len(test_data))
#    for i in range(0,len(test_data)):
#        test_labels[i] = test_data[i][2]
#    true_labels = np.ones(len(data))
#    for i in range(0,len(data)):
#        true_labels[i] = data[i][2]
#    no_rt_labels = np.ones(len(no_rt_data))
#    for i in range(0,len(no_rt_data)):
#        no_rt_labels[i] = no_rt_data[i][2]
    
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
#    pca.setup(features, 0.9)
#    features = pca.reduce(features)
#    test_features = pca.reduce(test_features)
#
#    pca2 = PCA()
#    pca2.setup(nort_features, 0.9)
#    nort_features = pca.reduce(nort_features)
#    test_features2 = pca.reduce(test_features2)

    k = 3
    print("KNN: k =",k)
    knn_model = KNN(k)
    knn_model.fit(features, true_labels)
    ymodel = knn_model.predict(test_features)
    tp,tn,fn,fp = perf_eval(ymodel, test_labels)
    print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))

    knn_model2 = KNN(k)
    knn_model2.fit(nort_features, nort_train_labels)
    ymodel = knn_model2.predict(test_features2)
    tp,tn,fn,fp = perf_eval(ymodel, test_labels)
    print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))

    print("MPP case 1")
    mpp = MPP(1)
#    mpp.set_prior(5/6, 1/6)
    mpp.fit(features, true_labels)
    ymodel = mpp.predict(test_features)
    tp,tn,fn,fp = perf_eval(ymodel, test_labels)
    print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))

    print("MPP case 2")
    mpp = MPP(2)
#    mpp.set_prior(5/6, 1/6)
    mpp.fit(features, true_labels)
    ymodel = mpp.predict(test_features)
    tp,tn,fn,fp = perf_eval(ymodel, test_labels)
    print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))

    print("MPP case 3")
    mpp = MPP(3)
#    mpp.set_prior(5/6, 1/6)
    mpp.fit(features, true_labels)
    ymodel = mpp.predict(test_features)
    tp,tn,fn,fp = perf_eval(ymodel, test_labels)
    print('Accuracy:     ', (tp+tn)/(tp+tn+fp+fn))

if __name__ == "__main__":
    main()
