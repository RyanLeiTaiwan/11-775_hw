import cv2
import numpy as np
import sys
import os
import fnmatch
import cPickle

# Extract SIFT features and draw keypoints for each keyframe
# http://docs.opencv.org/3.2.0/da/df5/tutorial_py_sift_intro.html
# http://docs.opencv.org/3.2.0/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Usage: {0} video_list keyframe_path sift_path keypoint_path'.format(sys.argv[0]))
        exit(1)

    video_list = open(sys.argv[1], 'r').read().splitlines()
    keyframe_path = sys.argv[2]
    sift_path = sys.argv[3]
    sift_suffix = '.sift.pk'
    keypoint_path = sys.argv[4]

    # Parameter: number of keypoints to detect
    param_nfeatures = 40
    print 'Extracting SIFT features...'
    #for video in ['HVC51']:
    for video in video_list:
        # Generate one matrix per video
        X = []
        kp_count = 0
        pathname = keyframe_path + '/' + video
        # Look for only jpg files
        frames = fnmatch.filter(os.listdir(pathname), '*.jpg')
        frames.sort()
        print 'video ' + video + ': ' + str(len(frames)) + ' keyframes'
        for frame in frames:
            inputName = pathname + '/' + frame
            img = cv2.imread(inputName)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # nfeatures: the number of best features to retain
            sift = cv2.xfeatures2d.SIFT_create(nfeatures=param_nfeatures)

            # kp: keypoints, des: descriptors
            # des.shape = (Number_of_Keypoints, 128)
            kp, des = sift.detectAndCompute(gray, None)
            kp_count += len(kp)
            # print '  keyframe ' + frame + ': ' + str(len(kp)) + ' keypoints'

            # Skip frames without any keypoints
            if des is not None:
                # Vertically concatenate the X matrix
                X += list(des)
                # If we want to draw and output keypoint images
                kpName = keypoint_path + '/' + video + '/' + frame
                # print '  keypoint image saved to ' + kpName
                kpImg = None
                kpImg = cv2.drawKeypoints(img, kp, kpImg)
                cv2.imwrite(kpName, kpImg)

        X = np.array(X)
        assert(X.shape[0] == kp_count)
        outputName = sift_path + '/' + video + sift_suffix
        print 'shape: ' + str(X.shape) + ' -> ' + outputName
        fout = open(outputName, 'wb')
        cPickle.dump(X, fout)
        fout.close()

    print 'SIFT features generated successfully!'
