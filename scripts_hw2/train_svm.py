import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
import cPickle
import sys

# Train SVM and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print 'Usage: {0} event_name label_file feat_file model_file'.format(sys.argv[0])
        exit(1)

    event_name = sys.argv[1]
    label_file = sys.argv[2]
    feat_file = sys.argv[3]
    model_file = sys.argv[4]

    # SVM parameters
    param_C = 1.0
    param_probability = False
    param_decision_function_shape = 'ovr'

    # Set a fixed random seed so we can reproduce the results
    np.random.seed(11775)

    print 'Loading feature matrix ' + feat_file
    fin = open(feat_file, 'rb')
    # feature file is already a numpy matrix
    X = np.float_(cPickle.load(fin))
    print 'X.shape: ' + str(X.shape)
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
    print 'y.shape: ' + str(y.shape) + ', # positive samples: ' + str(sum(y == 1))

    print 'Training SVM...'
    svm = SVC(probability=param_probability, C=param_C,
              decision_function_shape=param_decision_function_shape)
    # preprocessing.scale(): Normalize each column into 0 mean, 1 variance
    svm.fit(preprocessing.scale(X), y)

    # Output model
    print 'Output SVM model to ' + model_file
    fout = open(model_file, 'wb')
    cPickle.dump(svm, fout)
    fout.close()
    print 'SVM trained and model output successfully for event ' + event_name
