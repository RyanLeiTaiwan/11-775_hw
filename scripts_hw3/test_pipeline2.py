import numpy as np
from sklearn import preprocessing
import cPickle
import sys

# Test Pipeline 2 and output score_results and classification_results

if __name__ == '__main__':
    if len(sys.argv) != 14:
        print 'Usage: {0} event_name mfcc_feat mfcc_model asr_feat asr_model imtraj_feat imtraj_model \
sift_feat sift_model cnn_feat cnn_model score_results classification_results'.format(sys.argv[0])
        exit(1)

    event_name = sys.argv[1]
    mfcc_feat = sys.argv[2]
    mfcc_model = sys.argv[3]
    asr_feat = sys.argv[4]
    asr_model = sys.argv[5]
    imtraj_feat = sys.argv[6]
    imtraj_model = sys.argv[7]
    sift_feat = sys.argv[8]
    sift_model = sys.argv[9]
    cnn_feat = sys.argv[10]
    cnn_model = sys.argv[11]
    score_results = sys.argv[12]
    classification_results = sys.argv[13]

    # Set a fixed random seed so we can reproduce the results
    np.random.seed(11775)
    # Highest MAP
    weights = {'mfcc': 0.44, 'asr': 0.21, 'imtraj': 0.37, 'sift': 0.28, 'cnn': 0.93}

    print 'Loading feature matrices:'
    print mfcc_feat
    fin = open(mfcc_feat, 'rb')
    X_mfcc = preprocessing.scale(np.float_(cPickle.load(fin)))
    # print 'X_mfcc.shape: ' + str(X_mfcc.shape)
    print asr_feat
    fin = open(asr_feat, 'rb')
    X_asr = preprocessing.scale(np.float_(cPickle.load(fin)))
    # print 'X_asr.shape: ' + str(X_asr.shape)
    print imtraj_feat
    fin = open(imtraj_feat, 'rb')
    X_imtraj = preprocessing.scale(np.float_(cPickle.load(fin)))
    # print 'X_imtraj.shape: ' + str(X_imtraj.shape)
    print sift_feat
    fin = open(sift_feat, 'rb')
    X_sift = preprocessing.scale(np.float_(cPickle.load(fin)))
    # print 'X_sift.shape: ' + str(X_sift.shape)
    print cnn_feat
    fin = open(cnn_feat, 'rb')
    X_cnn = preprocessing.scale(np.float_(cPickle.load(fin)))
    # print 'X_cnn.shape: ' + str(X_cnn.shape)
    fin.close()

    print 'Loading SVM model files:'
    print mfcc_model
    fin = open(mfcc_model)
    svm_mfcc = cPickle.load(fin)
    fin.close()
    print asr_model
    fin = open(asr_model)
    svm_asr = cPickle.load(fin)
    fin.close()
    print imtraj_model
    fin = open(imtraj_model)
    svm_imtraj = cPickle.load(fin)
    fin.close()
    print sift_model
    fin = open(sift_model)
    svm_sift = cPickle.load(fin)
    fin.close()
    print cnn_model
    fin = open(cnn_model)
    svm_cnn = cPickle.load(fin)
    fin.close()

    print 'Ensemble testing...'
    # Linear combination of prediction scores
    y_score = weights['mfcc'] * svm_mfcc.decision_function(X_mfcc)
    y_score += weights['asr'] * svm_asr.decision_function(X_asr)
    y_score += weights['imtraj'] * svm_imtraj.decision_function(X_imtraj)
    y_score += weights['sift'] * svm_sift.decision_function(X_sift)
    y_score += weights['cnn'] * svm_cnn.decision_function(X_cnn)
    # Convert scores into class labels
    y_predict = np.int_(y_score > 0)
    # print 'y_score.shape: ' + str(y_score.shape)
    # print 'y_score:\n' + str(y_score)
    # print 'y_predict.shape: ' + str(y_predict.shape)
    # print 'y_predict:\n' + str(y_predict)

    print 'Output score_results and classification_results'
    outputName = score_results + '/' + event_name + '_score.txt'
    np.savetxt(outputName, y_score, fmt='%.17f')
    outputName = classification_results + '/' + event_name + '_class.txt'
    np.savetxt(outputName, y_predict, fmt='%d')

    print 'Pipeline 2 tested and results output successfully for event ' + event_name
