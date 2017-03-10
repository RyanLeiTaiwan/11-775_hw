#!/bin/python
import numpy as np
from sklearn.feature_extraction.text import *
import sys
import cPickle

# Train bag-of-word (word count) model

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print 'Usage: {0} asrtxt_path video_list vocab_file model_file'.format(sys.argv[0])
        print 'asrtxt_path -- path to the ASR transcribed text files'
        print 'video_list -- the list of videos'
        print 'vocab_file -- path to vocabulary file (output)'
        print 'model_file -- path to BoW model file (output)'
        exit(1)

    asrtxt_path = sys.argv[1]
    video_list = open(sys.argv[2]).read().splitlines()
    vocab_file = sys.argv[3]
    model_file = sys.argv[4]
    txt_suffix = '.txt'
    pk_suffix = '.pk'

    rawtxt = []
    # BoW representation: TF
    # "Noisewords" produced by EESEN ASR tool
    # noisewords = ['breath', 'cough', 'smack', 'um']
    model = CountVectorizer()
    # model = CountVectorizer(stop_words=noisewords + list(ENGLISH_STOP_WORDS))
    # model = TfidfVectorizer(stop_words=noisewords + list(ENGLISH_STOP_WORDS))

    # For each video
    for video in video_list:
        asrtxt_filename = asrtxt_path + "/" + video + txt_suffix
        # print asrtxt_filename
        asrtxt_file = open(asrtxt_filename)
        rawtxt.append(asrtxt_file.read())
        asrtxt_file.close()

    # Train Bag-of-Word representation
    BoW = model.fit_transform(rawtxt)
    X = BoW.toarray()
    vocab = model.get_feature_names()
    # print zip(range(len(vocab)), vocab)

    print 'vocabulary size: ' + str(len(vocab))
    print 'stop words: ' + str(model.stop_words_)
    print 'word count matrix.shape: ' + str(X.shape)

    # Output BoW model
    print 'Output BoW model to ' + model_file
    fout = open(model_file, 'wb')
    cPickle.dump(model, fout)
    fout.close()

    # Also output the vocab file
    print 'Output vocabulary to ' + vocab_file
    fout = open(vocab_file, 'w')
    fout.write(str(zip(range(len(vocab)), vocab)))
    fout.close()
    print "BoW model trained and output successfully!"
