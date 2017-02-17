#!/bin/python

import numpy as np
# import os
from sklearn.cluster import MiniBatchKMeans
import cPickle
import sys

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':

    if len(sys.argv) != 6:
        print "Usage: {0} cluster_num video_list feat_path feat_suffix model_path".format(sys.argv[0])
        exit(1)

    cluster_num = sys.argv[1]
    video_list = open(sys.argv[2]).read().splitlines()
    feat_path = sys.argv[3]
    feat_suffix = sys.argv[4]
    model_path = sys.argv[5]

    batch_size = 1000
    n_init = 3
    verbose = True

    # Set a fixed random seed so we can reproduce the results
    np.random.seed(11775)
    print 'Loading Feature Pickles...'
    X_all = []
    for video in video_list:
        inputName = feat_path + '/' + video + feat_suffix
        fin = open(inputName, 'rb')
        X_single = cPickle.load(fin)
        fin.close()
        print 'video: ' + video + ', shape: ' + str(X_single.shape)
        # Vertically append X_single to X_all
        X_all += list(X_single)

    X_all = np.array(X_all)
    print 'X_all.shape: ' + str(X_all.shape)

    exit(0)

    print 'cluster_num: ' + str(cluster_num)
    kmeans = MiniBatchKMeans(n_clusters=cluster_num, batch_size=batch_size,
                             init="k-means++", n_init=n_init, verbose=verbose)
    kmeans.fit(X_all)
    print 'K-means inertia: ' + str(kmeans.inertia_) + "\n"
    print 'k_means.cluster_centers_.shape: ' + str(kmeans.cluster_centers_.shape)

    # Output the K-means model
    outputName = model_path + '/kmeans.' + str(cluster_num) + '.model'
    fout = open(outputName, 'wb')
    cPickle.dump(kmeans, fout)
    fout.close()

    print 'K-means model trained and output successfully!'
