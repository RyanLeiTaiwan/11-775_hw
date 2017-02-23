import numpy as np
from sklearn.decomposition import PCA
import cPickle
import sys

# Run PCA to reduce dimensionality of imtraj feature

if __name__ == '__main__':
    # Use 'mode' argument to determine training / testing mode
    if len(sys.argv) != 7:
        print 'Usage: {0} mode video_list feat_dir feat_dim pca_file pca_model'.format(sys.argv[0])
        print 'mode: "train" (pca_model as output) or "test" (pca_model as input)'
        exit(1)
    if sys.argv[1] == 'train':
        mode = 'TRAINING'
    elif sys.argv[1] == 'test':
        mode = 'TESTING'
    else:
        print 'Invalid PCA mode!'
        print 'Usage: {0} mode video_list feat_dir feat_dim pca_file pca_model'.format(sys.argv[0])
        print 'mode: "train" (pca_model as output) or "test" (pca_model as input)'

    feat_suffix = '.spbof'
    video_list = open(sys.argv[2]).read().splitlines()
    feat_dir = sys.argv[3]
    feat_dim = int(sys.argv[4])
    pca_file = sys.argv[5]
    pca_model = sys.argv[6]

    # Set a fixed random seed so we can reproduce the results
    np.random.seed(11775)

    print '====  PCA ' + mode + ' mode  ===='
    # Testing mode: Load PCA model
    if mode == 'TESTING':
        print 'Loading existing PCA model ' + pca_model
        fmodel = open(pca_model, 'rb')
        pca = cPickle.load(fmodel)
        fmodel.close()

    print 'Loading sparse feature files...'
    X = []
    # for video in ['HVC51', 'HVC72', 'HVC285']:
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

    if mode == 'TRAINING':
        print 'Running PCA training to examine the variance explained...'
        # Examine the % of variance explained by principal components
        pca = PCA()
        pca.fit(X)
        var = np.cumsum(pca.explained_variance_ratio_)
        for ratio in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99]:
            print str(ratio) + ' variance explained by ' + str(sum(var < ratio) + 1) + ' components'
        # Set the target dimensionality to the one retaining 0.99 variance
        pca_dim = sum(var < 0.99) + 1
        print 'target dimensionality: ' + str(pca_dim)

        # Reduce dimensionality (sklearn requires us to 'fit' again after resetting n_components)
        print 'Running PCA training again + testing to reduce dimensionality...'
        pca.n_components = pca_dim
        X_pca = pca.fit_transform(X)

        # Training mode: also output PCA model file
        fmodel = open(pca_model, 'wb')
        cPickle.dump(pca, fmodel)
        fmodel.close()

    else:
        # Testing mode: Should NOT 'fit' again
        print 'Running PCA testing to reduce dimensionality...'
        X_pca = pca.transform(X)

    print "X_pca.shape: " + str(X_pca.shape)

    # Output reduced feature as a single pickle
    print 'Output reduced feature as a single pickle: ' + pca_file
    fout = open(pca_file, 'wb')
    cPickle.dump(X_pca, fout)
    fout.close()
