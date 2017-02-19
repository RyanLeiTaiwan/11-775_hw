import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix
import cPickle
import sys

# Report 3-fold cross-validation scores (MAP, class accuracy, TP rate TN rate)
# This program should only output the 4 metric scores to stdout

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print 'Usage: {0} event_name label_file feat_file model_file'.format(sys.argv[0])
        exit(1)

    event_name = sys.argv[1]
    label_file = sys.argv[2]
    feat_file = sys.argv[3]
    model_file = sys.argv[4]

    # Set a fixed random seed so we can reproduce the results
    np.random.seed(11775)

    # Load SVM model file
    fin = open(model_file)
    svm = cPickle.load(fin)
    fin.close()

    # Load feature matrix
    fin = open(feat_file, 'rb')
    # feature file is already a numpy matrix
    X = np.float_(cPickle.load(fin))
    # print 'X.shape: ' + str(X.shape)
    fin.close()

    # Load label file
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
    # preprocessing.scale(): Normalize each column into 0 mean, 1 variance
    X_scaled = preprocessing.scale(X)

    # Perform 3-fold cross-validation to report 4 metrics
    MAP = []
    ACC = []
    TPR = []
    TNR = []
    kf = KFold(n_splits=3, shuffle=True)
    # This loop will run for 3 iterations
    for train_idx, valid_idx in kf.split(X_scaled):
        # Split data into train : valid = 2 : 1
        X_train = X_scaled[train_idx, :]
        y_train = y[train_idx]
        X_valid = X_scaled[valid_idx, :]
        y_valid = y[valid_idx]

        # Train model and predict
        svm.fit(X_train, y_train)
        y_predict = svm.predict(X_valid)
        # print y_predict
        # print y_valid

        # Compute the metrics
        # [1] MAP
        MAP.append(average_precision_score(y_valid, y_predict))
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
    print 'CLASS ACCURACY: ' + str(np.mean(ACC))
    print 'TRUE POSITIVE RATE: ' + str(np.mean(TPR))
    print 'TRUE NEGATIVE RATE: ' + str(np.mean(TNR))
