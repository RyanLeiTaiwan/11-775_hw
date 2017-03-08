import numpy as np
import cPickle
import sys

# Assign K-means clusters to create BoW features
# Each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 7:
        print "Usage: {0} cluster_num video_list feat_path feat_suffix model_file BoW_file".format(sys.argv[0])
        exit(1)

    cluster_num = int(sys.argv[1])
    video_list = open(sys.argv[2]).read().splitlines()
    feat_path = sys.argv[3]
    feat_suffix = sys.argv[4]
    model_file = sys.argv[5]
    output_file = sys.argv[6]

    print 'Loading K-means model: ' + model_file
    fin = open(model_file, 'rb')
    kmeans = cPickle.load(fin)
    kmeans.verbose = False
    fin.close()

    print 'Clustering frames and computing BoW features...'
    X_all = []
    # for video in ['HVC51']:
    for video in video_list:
        # Load the feature file of all the frames
        feat_file = feat_path + '/' + video + feat_suffix
        fin = open(feat_file)
        X_single = cPickle.load(fin)
        fin.close()
        print '  ' + feat_file + ', shape: ' + str(X_single.shape)
        # Predict cluster labels
        labels = kmeans.predict(X_single)
        assert(len(labels) == X_single.shape[0])
        # print 'labels.shape: ' + str(labels.shape)
        # print labels

        # Count frequencies and output BoW features
        BoW = np.bincount(labels, minlength=cluster_num)
        assert(len(BoW) == cluster_num)
        # print 'BoW.shape: ' + str(BoW.shape)
        # print BoW

        X_all.append(BoW)

    X_all = np.array(X_all)
    print 'X_all.shape: ' + str(X_all.shape)
    assert(X_all.shape[0] == len(video_list))
    print 'Output BoW features to ' + output_file
    fout = open(output_file, 'wb')
    cPickle.dump(X_all, fout)
    fout.close()

    print 'K-means BoW features generated successfully!'
