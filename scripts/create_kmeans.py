#!/bin/python
import numpy as np
# import os
import cPickle
# from sklearn.cluster import KMeans
import sys
# Generate k-means features for videos;
# each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0])
        print "kmeans_model -- path to the kmeans model"
        print "cluster_num -- number of cluster"
        print "file_list -- the list of videos"
        exit(1)

    kmeans_model = open(sys.argv[1], "rb")
    cluster_num = int(sys.argv[2])
    file_list = sys.argv[3]
    mfcc_path = "mfcc/"
    mfcc_suffix = ".mfcc.csv"
    output_path = "kmeans/"
    BoW_suffix = ".bow.csv"

    # Load K-means model
    print "Load K-means model: " + kmeans_model
    kmeans = cPickle.load(kmeans_model)

    # Load file list
    print "Load file list"
    files = open(file_list, "r").read().splitlines()
    files.close()

    # For each video
    for video in files:
        mfcc_csv_file = mfcc_path + video + mfcc_suffix
        # Load the MFCC file of all the frames
        mfcc_matrix = np.loadtxt(mfcc_csv_file, delimiter=";")
        print mfcc_csv_file + ": " + str(mfcc_matrix.shape)
        # Predict cluster labels
        labels = kmeans.predict(mfcc_matrix)

        # Count frequencies and output BoW features
        BoW = np.array([np.bincount(labels, minlength=cluster_num)])
        print "BoW.shape: " + str(BoW.shape)
        np.savetxt(output_path + video + BoW_suffix, BoW, fmt="%d", delimiter=";")

    print "K-means features generated successfully!"
