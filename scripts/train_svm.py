#!/bin/python

import numpy as np
# import os
# from sklearn.svm.classes import SVC
from sklearn.svm import SVC
from sklearn import preprocessing
import cPickle
import sys

# Train SVM and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} event_name feat_dir output_file".format(sys.argv[0])
        print "event_name -- name of the event (P001, P002 or P003 in Homework 1)"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features"
        print "output_file -- path to save the svm model"
        exit(1)

    event_name = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]
    feat_suffix = ".bow.csv"
    
    # Set a fixed random seed so we can reproduce the results
    np.random.seed(11775)

    # Load the training list file
    list_filename = "list/" + event_name + "_train"
    print "Load the list file: " + list_filename
    list_file = open(list_filename).read().splitlines()
    X = []
    y = []
    for line in list_file:
        tok = line.split(" ")
        X_filename = feat_dir + "/" + tok[0] + feat_suffix
        # print X_filename
        data = np.loadtxt(X_filename, delimiter=";").tolist()
        assert(len(data) == feat_dim)
        X.append(data)

        # event_name: y = 1; NULL: y = 0
        if tok[1] == event_name:
            y.append(1)
        else:
            y.append(0)

    X = np.array(X)
    y = np.array(y)
    print "X.shape: " + str(X.shape)
    print "y.shape: " + str(y.shape) + ", positive samples: " + str(sum(y == 1))

    # Train SVM
    # print "Training SVM"
    svm = SVC(probability=True)
    # Normalize each column into 0 mean, 1 variance
    svm.fit(preprocessing.scale(X), y)

    # Output model
    fout = open(output_file, "wb")
    cPickle.dump(svm, fout)
    fout.close()

    print "SVM trained and model output successfully for event " + event_name
