#!/bin/bash

# Extract SIFT features and draw keypoints for each keyframe

#video_list=../list_hw2/train_dev.video
video_list=../list_hw2/test_hw2
keyframe_path=../keyframe
sift_path=../sift
keypoint_path=../keypoint

echo "Creating SIFT and keypoint directories..."
mkdir -p $sift_path
for file in $(cat $video_list)
do
    mkdir -p $keypoint_path/$file
done
python2 sift.py $video_list $keyframe_path $sift_path $keypoint_path

