import sys
import numpy as np
import cPickle

# Combine CNN features of individual videos into one matrix of shape (num_videos, 4096)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: {0} video_list cnn_individual_path cnn_combine_file'.format(sys.argv[0]))
        exit(1)

    video_list = open(sys.argv[1], 'r').read().splitlines()
    cnn_individual_path = sys.argv[2]
    cnn_combine_file = sys.argv[3]

    print "Combining CNN features into one file..."
    suffix = '.pk'

    # Loop through each video's keyframe
    X_all = []
    # for video in ['HVC51', 'HVC72', 'HVC285', 'HVC286', 'HVC288']:
    for video in video_list:
        X_single = []
        filename = cnn_individual_path + '/' + video + suffix
        print '  ' + filename
        fin = open(filename, 'rb')
        X_single = cPickle.load(fin)
        fin.close()
        assert(X_single.shape == (4096,))
        X_all.append(list(X_single))

    X_all = np.array(X_all)
    assert(X_all.shape[0] == len(video_list))
    print 'X_all.shape: ' + str(X_all.shape)

    # Output as cPickle
    fout = open(cnn_combine_file, 'wb')
    cPickle.dump(X_all, fout)
    fout.close()

    print 'CNN features combined into ' + cnn_combine_file
