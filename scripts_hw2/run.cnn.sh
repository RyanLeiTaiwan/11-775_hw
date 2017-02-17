#!/bin/bash

# Extract Cafee CNN features for each keyframe

#video_list=../list_hw2/train_dev.video
video_list=../list_hw2/test_hw2
keyframe_path=../keyframe
cnn_path=../cnn_pred
cnn_file=test

mkdir -p $cnn_path
python2 cnn.py $video_list $keyframe_path $cnn_path $cnn_file

