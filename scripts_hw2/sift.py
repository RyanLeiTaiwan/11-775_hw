import cv2
import numpy as np
import os
# from sklearn.cluster import KMeans
import sys

# Generate SIFT features for each keyframe
# http://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: {0} video_list keyframe_path keypoint_path'.format(sys.argv[0]))
        exit(1)

    video_list = open(sys.argv[1], 'r').read().splitlines()
    keyframe_path = sys.argv[2]
    sift_suffix = '.sift'
    keypoint_path = sys.argv[3]

    print 'Extracting SIFT features...'
    # for video in ['HVC51']:
    for video in video_list:
        pathname = keyframe_path + '/' + video
        for frame in os.listdir(pathname):
            # Skip non-jpg files
            if frame.endswith('.jpg'):
                inputName = pathname + '/' + frame
                print 'Input: ' + inputName

                img = cv2.imread(inputName)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sift = cv2.xfeatures2d.SIFT_create()

                # keypoint coordinates and descriptors
                kp, des = sift.detectAndCompute(gray, None)
                # Skip frames without any keypoints
                if des is not None:
                    outputName = pathname + '/' + frame.split('.')[0] + sift_suffix
                    print 'Output: ' + outputName
                    np.savetxt(outputName, des, fmt='%.6e', delimiter=';')

                    # If we want to draw and output keypoint images
                    kpName = keypoint_path + '/' + video + '/' + frame
                    print 'Keypoint: ' + kpName
                    kpImg = None
                    kpImg = cv2.drawKeypoints(img, kp, kpImg)
                    cv2.imwrite(kpName, kpImg)
    print 'SIFT features generated successfully!'
