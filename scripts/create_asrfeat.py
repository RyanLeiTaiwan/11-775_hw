#!/bin/python
import numpy as np
from sklearn.feature_extraction.text import *
import sys

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} asrtxt_path file_list vocab_file asrfeat_dir".format(sys.argv[0])
        print "asrtxt_path -- path to the ASR transcribed text files"
        print "file_list -- the list of videos"
        print "vocab_file -- path to the vocabulary file (output)"
        print "asrfeat_dir -- path to the ASR feature files (output)"
        exit(1)

    asrtxt_path = sys.argv[1]
    file_list = open(sys.argv[2], "r")
    vocab_file = open(sys.argv[3], "w")
    asrfeat_dir = sys.argv[4]
    asrtxt_suffix = ".txt"
    BoW_suffix = ".bow.csv"

    # For each video
    rawtxt = []
    # BoW representation: TF
    model = CountVectorizer()
    #model = TfidfVectorizer()
    files = file_list.read().splitlines()
    file_list.close()
    for video in files:
        asrtxt_filename = asrtxt_path + "/" + video + asrtxt_suffix
        # print asrtxt_filename
        asrtxt_file = open(asrtxt_filename)
        rawtxt.append(asrtxt_file.read())
        asrtxt_file.close()

    # Build Bag-of-Word representation
    BoW = model.fit_transform(rawtxt)
    X = BoW.toarray()
    vocab = model.vocabulary_

    print "vocabulary size: " + str(len(model.vocabulary_))
    print "stop words: " + str(model.stop_words_)
    print "matrix X.shape: " + str(X.shape)

    # To match MFCC's organization, we should output a matrix row for each video
    feat_dim = X.shape[1]
    for row in range(X.shape[0]):
        output_filename = asrfeat_dir + "/" + files[row] + BoW_suffix
        print "Saving ASR features to " + output_filename
        np.savetxt(output_filename, X[row, :].reshape(1, feat_dim), fmt="%d", delimiter=";")

    # Also output the vocab file
    vocab_file.write(str(vocab))
    vocab_file.close()
    print "ASR features generated successfully!"
