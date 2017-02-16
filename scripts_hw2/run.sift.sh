#!/bin/bash

# Extract SIFT features for each keyframe

video_list=../list_hw2/train_dev.video
keyframe_path=../keyframe
keypoint_path=../keypoint

echo "Creating SIFT and keypoint directories..."
for file in $(cat $video_list)
do
    mkdir -p $keypoint_path/$file
done
python2 sift.py $video_list $keyframe_path $keypoint_path

