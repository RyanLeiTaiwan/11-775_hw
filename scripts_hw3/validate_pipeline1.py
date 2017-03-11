import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix
from sklearn import base
import cPickle
import sys

# MED Pipeline 1: Linear combination of prediction scores [Uniform weights]

if __name__ == '__main__':
    if len(sys.argv) != 8:
        print 'Usage: {0} event_name label_file mfcc_feat asr_feat imtraj_feat sift_feat cnn_feat'.format(sys.argv[0])
        exit(1)

    event_name = sys.argv[1]
    label_file = sys.argv[2]
    mfcc_feat = sys.argv[3]
    asr_feat = sys.argv[4]
    imtraj_feat = sys.argv[5]
    sift_feat = sys.argv[6]
    cnn_feat = sys.argv[7]
    # Set a fixed random seed so we can reproduce the results
    np.random.seed(11775)
    # VotingClassifier weights
    # Uniform weights
    weights = {'mfcc': 1.00, 'asr': 1.00, 'imtraj': 1.00, 'sift': 1.00, 'cnn': 1.00}
    # P001 MAP
    # weights = {'mfcc': 0.33, 'asr': 0.19, 'imtraj': 0.28, 'sift': 0.28, 'cnn': 0.75}
    # P002 MAP
    # weights = {'mfcc': 0.44, 'asr': 0.06, 'imtraj': 0.37, 'sift': 0.27, 'cnn': 0.93}
    # # P003 MAP
    # weights = {'mfcc': 0.17, 'asr': 0.21, 'imtraj': 0.27, 'sift': 0.20, 'cnn': 0.52}
    # # Highest MAP
    # weights = {'mfcc': 0.44, 'asr': 0.21, 'imtraj': 0.37, 'sift': 0.28, 'cnn': 0.93}
    # # Highest TPR
    # weights = {'mfcc': 0.04, 'asr': 0.02, 'imtraj': 0.02, 'sift': 0.02, 'cnn': 0.74}
    # # P002 TPR
    # weights = {'mfcc': 0.00, 'asr': 0.00, 'imtraj': 0.00, 'sift': 0.00, 'cnn': 0.74}

    # SVM parameters
    param_C = 1.0
    param_probability = False
    param_decision_function_shape = 'ovr'
    param_random_state = 11775

    # print 'Loading feature matrices:'
    # print mfcc_feat
    fin = open(mfcc_feat, 'rb')
    X_mfcc = preprocessing.scale(np.float_(cPickle.load(fin)))
    # print 'X_mfcc.shape: ' + str(X_mfcc.shape)
    # print asr_feat
    fin = open(asr_feat, 'rb')
    X_asr = preprocessing.scale(np.float_(cPickle.load(fin)))
    # print 'X_asr.shape: ' + str(X_asr.shape)
    # print imtraj_feat
    fin = open(imtraj_feat, 'rb')
    X_imtraj = preprocessing.scale(np.float_(cPickle.load(fin)))
    # print 'X_imtraj.shape: ' + str(X_imtraj.shape)
    # print sift_feat
    fin = open(sift_feat, 'rb')
    X_sift = preprocessing.scale(np.float_(cPickle.load(fin)))
    # print 'X_sift.shape: ' + str(X_sift.shape)
    # print cnn_feat
    fin = open(cnn_feat, 'rb')
    X_cnn = preprocessing.scale(np.float_(cPickle.load(fin)))
    # print 'X_cnn.shape: ' + str(X_cnn.shape)
    fin.close()

    # print 'Loading label file ' + label_file
    y = []
    fin = open(label_file, 'r')
    for label in fin.read().splitlines():
        # event_name: y = 1; otherwise: y = 0
        if label == event_name:
            y.append(1)
        else:
            y.append(0)
    y = np.array(y)
    # print 'y.shape: ' + str(y.shape) + ', # positive samples: ' + str(sum(y == 1))

    # print 'Ensemble cross-validation...'
    MAP = []
    ACC = []
    TPR = []
    TNR = []
    kf = KFold(n_splits=3, shuffle=True)
    # This loop will run for 3 iterations
    for train_idx, valid_idx in kf.split(y):
        # Split data into train : valid = 2 : 1
        X_mfcc_train = X_mfcc[train_idx, :]
        X_mfcc_valid = X_mfcc[valid_idx, :]
        X_asr_train = X_asr[train_idx, :]
        X_asr_valid = X_asr[valid_idx, :]
        X_imtraj_train = X_imtraj[train_idx, :]
        X_imtraj_valid = X_imtraj[valid_idx, :]
        X_sift_train = X_sift[train_idx, :]
        X_sift_valid = X_sift[valid_idx, :]
        X_cnn_train = X_cnn[train_idx, :]
        X_cnn_valid = X_cnn[valid_idx, :]
        y_train = y[train_idx]
        y_valid = y[valid_idx]

        # Train model and predict
        svm_mfcc = SVC(probability=param_probability, C=param_C,
                       decision_function_shape=param_decision_function_shape,
                       random_state=param_random_state)
        svm_asr = base.clone(svm_mfcc)
        svm_imtraj = base.clone(svm_mfcc)
        svm_sift = base.clone(svm_mfcc)
        svm_cnn = base.clone(svm_mfcc)

        svm_mfcc.fit(X_mfcc_train, y_train)
        svm_asr.fit(X_asr_train, y_train)
        svm_imtraj.fit(X_imtraj_train, y_train)
        svm_sift.fit(X_sift_train, y_train)
        svm_cnn.fit(X_cnn_train, y_train)

        # Linear combination of prediction scores
        y_score = weights['mfcc'] * svm_mfcc.decision_function(X_mfcc_valid)
        y_score += weights['asr'] * svm_asr.decision_function(X_asr_valid)
        y_score += weights['imtraj'] * svm_imtraj.decision_function(X_imtraj_valid)
        y_score += weights['sift'] * svm_sift.decision_function(X_sift_valid)
        y_score += weights['cnn'] * svm_cnn.decision_function(X_cnn_valid)
        # Convert scores into class labels
        y_predict = np.int_(y_score > 0)
        # print 'y_predict:\n' + str(y_predict)
        # print 'y_valid:\n' + str(y_valid)

        # Compute the metrics
        # [1] MAP
        MAP.append(average_precision_score(y_valid, y_score))
        # [2] Class Accuracy
        ACC.append(accuracy_score(y_valid, y_predict))

        # Use confusion matrix to compute TPR and TNR
        conf = confusion_matrix(y_valid, y_predict)

        # [3] True Positive Rate (TPR)
        TPR.append(np.float_(conf[1, 1]) / np.float_(sum(conf[1, ])))

        # [4] True Negative Rate (TNR)
        TNR.append(np.float_(conf[0, 0]) / np.float_(sum(conf[0, ])))

    # Display average results
    print 'MAP: ' + str(np.mean(MAP))
    print 'ACCURACY: ' + str(np.mean(ACC))
    print 'TPR: ' + str(np.mean(TPR))
    print 'TNR: ' + str(np.mean(TNR))
