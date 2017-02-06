#!/bin/python

import numpy as np
# import os
# from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import cPickle
import sys

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Usage: {0} mfcc_csv_file cluster_num output_file".format(sys.argv[0])
        print "mfcc_csv_file -- path to the mfcc csv file"
        print "output_file -- path to save the k-means model"
        exit(1)

    mfcc_csv_file = sys.argv[1]
    output_folder = sys.argv[2]

    # Set a fixed random seed so we can reproduce the results
    np.random.seed(11775)
    print "Loading MFCC CSV file"
    data = np.loadtxt(mfcc_csv_file, delimiter=";")
    print "data.shape: " + str(data.shape)

    for cluster_num in range(50, 501, 50):
        print "cluster_num: " + str(cluster_num)
        # kmeans = KMeans(n_clusters=cluster_num, init="k-means++", n_init=10)
        kmeans = MiniBatchKMeans(n_clusters=cluster_num, batch_size=10000, init="k-means++", n_init=3)
        kmeans.fit(data)
        # print kmeans
        print "K-means inertia: " + str(kmeans.inertia_) + "\n"
        # print "k_means.cluster_centers_.shape: " + str(kmeans.cluster_centers_.shape)

        # Output the K-means model
        output_filename = output_folder + "/kmeans." + str(cluster_num) + ".model"
        output_file = open(output_filename, "wb")
        cPickle.dump(kmeans, output_file)
    print "K-means model trained and output successfully!"

