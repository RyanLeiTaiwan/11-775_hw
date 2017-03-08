import sys
import numpy as np
import cPickle

# Load CSV files and convert into pickle files to comply with K-means input format

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print('Usage: {0} video_list csv_path csv_suffix pk_path pk_suffix'.format(sys.argv[0]))
        exit(1)

    video_list = open(sys.argv[1], 'r').read().splitlines()
    csv_path = sys.argv[2]
    csv_suffix = sys.argv[3]
    pk_path = sys.argv[4]
    pk_suffix = sys.argv[5]

    # Loop through each video's keyframe
    for video in ['HVC51', 'HVC72']:
    # for video in video_list:
        inputName = csv_path + '/' + video + csv_suffix
        outputName = pk_path + '/' + video + pk_suffix
        print inputName + ' -> ' + outputName
        X = np.loadtxt(inputName, delimiter=';')
        print 'X.shape: ' + str(X.shape)
        assert(X.shape[1] == 39)
        fout = open(outputName, 'wb')
        cPickle.dump(X, fout)
        fout.close()

    print 'Finished converting CSV into pickle'
