#!/bin/bash

# Extract SIFT features and draw keypoints for each keyframe

video_list=../list_hw3/all.video
keyframe_path=../keyframe_vfr
sift_path=../sift_vfr
keypoint_path=../keypoint_vfr

echo "Creating SIFT and keypoint directories..."
mkdir -p $sift_path
for file in $(cat $video_list)
do
    mkdir -p $keypoint_path/$file
done
python2 sift.py $video_list $keyframe_path $sift_path $keypoint_path

