#!/bin/bash

# Extract SIFT features for each keyframe

cut=30
video_list=../list_hw2/train_dev.video
keyframe_path=../keyframe_$cut
keypoint_path=../keypoint_$cut

#echo "Creating SIFT and keypoint directories..."
#for file in $(cat $video_list)
#do
#    mkdir -p $keypoint_path/$file
#done
python2 sift.py $video_list $keyframe_path $keypoint_path

