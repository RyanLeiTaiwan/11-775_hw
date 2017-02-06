#!/bin/python

import numpy as np
import os
from sklearn.svm import SVC
from sklearn import preprocessing
import cPickle
import sys

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0])
        print "model_file -- path of the trained svm file"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features; provided just for debugging"
        print "output_file -- path to save the prediction score"
        exit(1)

    model_file = open(sys.argv[1], "rb")
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]
    feat_suffix = ".bow.csv"

    # Load the SVM model
    svm = cPickle.load(model_file)
    model_file.close()

    # Load the testing list file
    X = []
    files = open("list/test.video").read().splitlines()
    for line in files:
        X_filename = feat_dir + "/" + line + feat_suffix
        # print X_filename
        data = np.loadtxt(X_filename, delimiter=";").tolist()
        assert(len(data) == feat_dim)
        X.append(data)

    X = np.array(X)
    print "X.shape: " + str(X.shape)

    # Test SVM
    print "Testing SVM"
    # Normalize each column into 0 mean, 1 variance
    y = svm.predict_proba(preprocessing.scale(X))[:, 1]
    print "y.shape: " + str(y.shape)

    # Output prediction labels
    np.savetxt(output_file, y)

    print "SVM tested and labels output successfully to " + output_file
