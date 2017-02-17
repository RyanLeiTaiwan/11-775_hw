import numpy as np
import caffe
import os
import sys
import fnmatch
import cPickle

# Extract Caffe CNN feature (4096-dim in fc7 layer) for each keyframe
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb
# https://prateekvjoshi.com/2016/04/26/how-to-extract-feature-vectors-from-deep-neural-networks-in-python-caffe/

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Usage: {0} video_list keyframe_path cnn_path cnn_file'.format(sys.argv[0]))
        exit(1)

    video_list = open(sys.argv[1], 'r').read().splitlines()
    keyframe_path = sys.argv[2]
    cnn_suffix = '.cnn.pk'
    cnn_path = sys.argv[3]
    cnn_file = sys.argv[4]

    print "Setting up Caffe CNN..."
    batch_size = 1
    layer = 'fc7'

    # Set up Caffe CNN
    caffe_root = '/home/ubuntu/tools/caffe/caffe/'
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
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))
    net.blobs['data'].reshape(batch_size, 3, 227, 227)
    

    print '\nExtracting Caffe CNN features...'
    # Only output one matrix of shape (#_of_videos, 4096)
    X_all = []
    # Loop through each video's keyframe
    #for video in ['HVC51']:
    for video in video_list:
        # Generate one "averaged" vector per video
        # TA: If the feature vector is sparse it is better to average rather than BoW
        X_single = []
        pathname = keyframe_path + '/' + video
        # Look for only jpg files
        frames = fnmatch.filter(os.listdir(pathname), '*.jpg')
        frames.sort()
        print 'video ' + video + ': ' + str(len(frames)) + ' keyframes'
        for frame in frames:
            inputName = pathname + '/' + frame
            image = caffe.io.load_image(inputName)
            transformed_image = transformer.preprocess('data', image)
            net.blobs['data'].data[...] = transformed_image
            # Run the forward procedure
            output = net.forward()
            # Extract the intermediate (fc7) layer as a numpy vector
            #print sum(net.blobs[layer].data[0] != 0)
            vector = net.blobs[layer].data[0]
            X_single.append(list(vector))
        # Video-level 
        X_single = np.array(X_single)
        assert(X_single.shape[0] == len(frames))
        # Average over the all frames (rows)
        X_single = np.mean(X_single, 0)
        # Append X_single vector to X_all matrix
        X_all.append(list(X_single))

    # Top-level
    X_all = np.array(X_all)
    assert(X_all.shape[0] == len(video_list))
    # Output as cPickle
    outputName = cnn_path + '/' + cnn_file + cnn_suffix
    print 'output matrix shape: ' + str(X_all.shape) + ' -> ' + outputName
    fout = open(outputName, 'wb')
    cPickle.dump(X_all, fout)
    fout.close()

    print 'Caffe CNN features extracted successfully!'

