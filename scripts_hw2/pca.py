#!/bin/python

import numpy as np
from sklearn.decomposition import PCA
import cPickle
import sys

# Run PCA to reduce dimensionality of imtraj feature

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print 'Usage: {0} video_list feat_dir feat_dim pca_path'.format(sys.argv[0])
        exit(1)

    video_list = open(sys.argv[1]).read().splitlines()
    feat_dir = sys.argv[2]
    feat_suffix = '.spbof'
    feat_dim = int(sys.argv[3])
    pca_path = sys.argv[4]

    # Set a fixed random seed so we can reproduce the results
    np.random.seed(11775)

    print 'Loading sparse feature files...'
    X = []
    # for video in ['HVC51']:
    for video in video_list:
        vec = [0] * feat_dim
        filename = feat_dir + '/' + video + feat_suffix
        print filename
        # Only one line per video
        feat_line = open(filename).readlines()[0].strip()
        # print line
        fields = feat_line.split(' ')
        for dim in fields:
            tok = dim.split(':')
            # feature file is 1-based indexing
            vec[int(tok[0]) - 1] = float(tok[1])
        X.append(vec)

    X = np.array(X)
    print "X.shape: " + str(X.shape)

    # # Temp: dump as pickle
    # ftmp = open('imtraj.pk', 'wb')
    # cPickle.dump(X, ftmp)

    # # Temp: load from pickle
    # ftmp = open('imtraj.pk', 'r')
    # X = cPickle.load(ftmp)
    # print 'Finished loading pk'

    print 'Running PCA to examine the variance explained...'
    # Examine the % of variance explained by principal components
    pca = PCA()
    pca.fit(X)
    var = np.cumsum(pca.explained_variance_ratio_)
    for ratio in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99]:
        print str(ratio) + ' variance explained by ' + str(sum(var < ratio) + 1) + ' components'
    # Set the target dimensionality to the one retaining 0.99 variance
    pca_dim = sum(var < 0.99) + 1
    print 'target dimensionality: ' + str(pca_dim)

    # Reduce dimensionality (sklearn requires us to 'fit' again)
    print 'Running PCA again to reduce dimensionality...'
    pca.n_components = pca_dim
    X_pca = pca.fit_transform(X)
    print "X_pca.shape: " + str(X_pca.shape)

    # Output reduced feature as a single pickle
    print 'Output reduced feature as a single pickle: ' + pca_path
    fout = open(pca_path, 'wb')
    cPickle.dump(X_pca, fout)
    fout.close()
