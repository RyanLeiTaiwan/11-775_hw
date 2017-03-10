#!/bin/python
import numpy as np
from sklearn.feature_extraction.text import *
import sys
import cPickle

# Convert plain text into a bag-of-word (word count) matrix

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print 'Usage: {0} asrtxt_path video_list vocab_file model_file'.format(sys.argv[0])
        print 'asrtxt_path -- path to the ASR transcribed text files'
        print 'video_list -- the list of videos'
        print 'model_file -- path to BoW model file (output)'
        print 'feat_file -- path to BoW feature file file (output)'
        exit(1)

    asrtxt_path = sys.argv[1]
    video_list = open(sys.argv[2]).read().splitlines()
    model_file = sys.argv[3]
    feat_file = sys.argv[4]
    txt_suffix = '.txt'

    # Load the BoW model
    fin = open(model_file, 'rb')
    model = cPickle.load(fin)

    rawtxt = []
    # For each video
    for video in video_list:
        asrtxt_filename = asrtxt_path + "/" + video + txt_suffix
        # print asrtxt_filename
        asrtxt_file = open(asrtxt_filename)
        rawtxt.append(asrtxt_file.read())
        asrtxt_file.close()

    # Test BoW model (word count matrix transform)
    BoW = model.transform(rawtxt)
    X = BoW.toarray()
    vocab = model.get_feature_names()
    # print zip(range(len(vocab)), vocab)

    print 'vocabulary size: ' + str(len(vocab))
    print 'stop words: ' + str(model.stop_words_)
    print 'word count matrix.shape: ' + str(X.shape)

    # Output BoW feature
    print 'Output BoW feature to ' + feat_file
    fout = open(feat_file, 'wb')
    cPickle.dump(X, fout)
    fout.close()
    print "BoW features generated successfully!"
