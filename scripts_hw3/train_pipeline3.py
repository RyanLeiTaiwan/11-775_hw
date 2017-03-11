import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import cPickle
import sys

# MED Pipeline 3: Two-level learning (SVMs + Logistic regression)

if __name__ == '__main__':
    if len(sys.argv) != 14:
        print 'Usage: {0} event_name label_file mfcc_feat mfcc_model asr_feat asr_model \
imtraj_feat imtraj_model sift_feat sift_model cnn_feat cnn_model LR_model'.format(sys.argv[0])
        exit(1)

    event_name = sys.argv[1]
    label_file = sys.argv[2]
    mfcc_feat = sys.argv[3]
    mfcc_model = sys.argv[4]
    asr_feat = sys.argv[5]
    asr_model = sys.argv[6]
    imtraj_feat = sys.argv[7]
    imtraj_model = sys.argv[8]
    sift_feat = sys.argv[9]
    sift_model = sys.argv[10]
    cnn_feat = sys.argv[11]
    cnn_model = sys.argv[12]
    LR_model = sys.argv[13]
    # Set a fixed random seed so we can reproduce the results
    np.random.seed(11775)

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

    print 'Loading label file ' + label_file
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

    print 'Ensemble training...'
    # Use SVM prediction scores (y) as 2nd-level features (X)
    y_mfcc = svm_mfcc.decision_function(X_mfcc)
    y_asr = svm_asr.decision_function(X_asr)
    y_imtraj = svm_imtraj.decision_function(X_imtraj)
    y_sift = svm_sift.decision_function(X_sift)
    y_cnn = svm_cnn.decision_function(X_cnn)

    X_LR = np.column_stack((y_mfcc, y_asr, y_imtraj, y_sift, y_cnn))
    print X_LR.shape

    # Train Logistic Regression classifier
    LR = LogisticRegression()
    LR.fit(X_LR, y)

    # Output LR model
    print 'Output Logistic Regression model to ' + LR_model
    fout = open(LR_model, 'wb')
    cPickle.dump(LR, fout)
    fout.close()
    print 'Logistic Regression trained and model output successfully for event ' + event_name
