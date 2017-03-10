import numpy as np
from sklearn import preprocessing
import cPickle
import sys

# Test SVM and output score_results and classification_results

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print 'Usage: {0} event_name feat_file model_file score_results classification_results'.format(sys.argv[0])
        exit(1)

    event_name = sys.argv[1]
    feat_file = sys.argv[2]
    model_file = sys.argv[3]
    score_results = sys.argv[4]
    classification_results = sys.argv[5]

    # Set a fixed random seed so we can reproduce the results
    np.random.seed(11775)

    print 'Loading SVM model file ' + model_file
    fin = open(model_file)
    svm = cPickle.load(fin)
    fin.close()

    print 'Loading feature matrix ' + feat_file
    fin = open(feat_file, 'rb')
    # feature file is already a numpy matrix
    X = np.float_(cPickle.load(fin))
    print 'X.shape: ' + str(X.shape)
    fin.close()

    print 'Testing SVM...'
    # preprocessing.scale(): Normalize each column into 0 mean, 1 variance
    X_scaled = preprocessing.scale(X)
    # y_score may be of 2 columns in future sklearn version
    # Depending on training, y_score may be svm.decision_function or svm.predict_proba[:, 1]

    y_score = svm.decision_function(X_scaled)
    # y_score = svm.predict_proba(X_scaled)[:, 1]
    y_label = svm.predict(X_scaled)

    print 'Output score_results and classification_results'
    outputName = score_results + '/' + event_name + '_score.txt'
    np.savetxt(outputName, y_score, fmt='%.17f')
    outputName = classification_results + '/' + event_name + '_class.txt'
    np.savetxt(outputName, y_label, fmt='%d')

    print 'SVM tested and results output successfully for event ' + event_name
