#!/bin/python

import numpy as np
# import os
from sklearn.cluster import KMeans
# from sklearn.cluster.k_means_ import KMeans
import cPickle
import sys

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print "Usage: {0} mfcc_csv_file cluster_num output_file".format(sys.argv[0])
        print "mfcc_csv_file -- path to the mfcc csv file"
        print "cluster_num -- number of cluster"
        print "output_file -- path to save the k-means model"
        exit(1)

    mfcc_csv_file = sys.argv[1]
    cluster_num = int(sys.argv[2])
    output_file = open(sys.argv[3], "wb")

    # Set a fixed random seed so we can reproduce the results
    np.random.seed(11775)
    data = np.loadtxt(mfcc_csv_file, delimiter=";")
    print "data.shape: " + str(data.shape)
    kmeans = KMeans(n_clusters=cluster_num, init="k-means++", n_init=10)
    kmeans.fit(data)
    # print kmeans
    print "K-means inertia: " + str(kmeans.inertia_)
    # print "k_means.cluster_centers_.shape: " + str(kmeans.cluster_centers_.shape)

    # Output the cluster centroids
    cPickle.dump(kmeans.cluster_centers_, output_file)
    print "K-means model trained and output successfully!"
