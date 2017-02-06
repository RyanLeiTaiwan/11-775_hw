#!/bin/python
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: {0} test_label pred_label".format(sys.argv[0])
        print "test_label -- path of ground truth labels of testing data"
        print "pred_label -- path of predicted labels of testing data"
        exit(1)

    # Load test data labels and predicted labels
    test_file = open(sys.argv[1], "r")
    test_label = test_file.read().splitlines()
    test_file.close()
    pred_file = open(sys.argv[2], "r")
    pred_label = pred_file.read().splitlines()
    pred_file.close()
    # print test_label
    # print pred_label

    # True positive, true negative, total_testing_samples
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    total = len(test_label)
    for sample in range(0, total):
        # Probability 0.5 is considered positive
        if int(test_label[sample]) == 1:
            if float(pred_label[sample]) >= 0.5:
                TP = TP + 1
            else:
                FN = FN + 1
        elif int(test_label[sample]) == 0:
            if float(pred_label[sample]) < 0.5:
                TN = TN + 1
            else:
                FP = FP + 1

    # Output class accuracy
    accuracy = float(TP + TN) / float(total)
    print "Class accuracy: " + str(accuracy) + " (" + str(TP + TN) + " / " + str(total) + ")"
    # print "TP: %d, FN: %d, TN: %d, FP: %d" % (TP, FN, TN, FP)
