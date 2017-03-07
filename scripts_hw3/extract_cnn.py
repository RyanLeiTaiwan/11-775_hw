import sys
import numpy as np
import os
import fnmatch
import cPickle

# Extract Caffe CNN feature (4096-dim in fc7 layer) for each keyframe
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb
# https://prateekvjoshi.com/2016/04/26/how-to-extract-feature-vectors-from-deep-neural-networks-in-python-caffe/

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: {0} video_list keyframe_path cnn_individual_path'.format(sys.argv[0]))
        exit(1)

    video_list = open(sys.argv[1], 'r').read().splitlines()
    keyframe_path = sys.argv[2]
    cnn_individual_path = sys.argv[3]

    print "Setting up Caffe CNN..."
    caffe_root = '/home/ubuntu/tools/caffe/'
    # Explicitly insert Python PATH
    sys.path.insert(0, caffe_root + 'python')
    import caffe
    batch_size = 1
    layer = 'fc7'
    output_suffix = '.pk'

    # Set up Caffe CNN
    if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
        print 'CaffeNet found.'
    else:
        print 'Error: pre-trained CaffeNet model not downloaded'
        exit(1)
    caffe.set_mode_cpu()
    model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
    model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)
    print 'mean-subtracted values:', zip('BGR', mu)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    net.blobs['data'].reshape(batch_size, 3, 227, 227)

    print '\nExtracting Caffe CNN features...'
    # Loop through each video's keyframe
    #for video in ['HVC51']:
    for video in video_list:
        # Output one "averaged" vector per video (1 x 4096)
        # TA: If the feature vector is sparse it is better to average rather than training BoW
        X = []
        pathname = keyframe_path + '/' + video
        # Look for only jpg files
        frames = fnmatch.filter(os.listdir(pathname), '*.jpg')
        frames.sort()
        print 'video ' + video + ': ' + str(len(frames)) + ' keyframes'
        for frame in frames:
            inputName = pathname + '/' + frame
            #print(inputName);
            image = caffe.io.load_image(inputName)
            transformed_image = transformer.preprocess('data', image)
            net.blobs['data'].data[...] = transformed_image
            # Run the forward procedure
            output = net.forward()
            # Extract the intermediate (fc7) layer as a numpy vector
            vector = net.blobs[layer].data[0]
            X.append(list(vector))
        # Video-level
        X = np.array(X)
        assert(X.shape[0] == len(frames))
        print 'Median zero entries before averaging: ' + str(np.median(np.sum(X == 0, 1)))
        # Average over the all frames (rows)
        X = np.mean(X, 0)
        print 'Zero entries after averaging: ' + str(np.sum(X == 0))
        outputName = cnn_individual_path + '/' + video + output_suffix
        #print 'X.shape: ' + str(X.shape) + ', output to ' + outputName

        # Output as cPickle
        fout = open(outputName, 'wb')
        cPickle.dump(X, fout)
        fout.close()

    print 'Caffe CNN features extracted successfully!'

