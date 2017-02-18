#!/bin/bash

# Extract Cafee CNN features for each keyframe

tain_video_list=../list_hw2/train_dev.video
test_video_list=../list_hw2/test_hw2
keyframe_path=../keyframe
cnn_path=../cnn_pred
train_cnn_file=$cnn_path/train.cnn.pk
test_cnn_file=$cnn_path/test.cnn.pk

mkdir -p $cnn_path
python2 cnn.py $train_video_list $keyframe_path $train_cnn_file
python2 cnn.py $test_video_list $keyframe_path $test_cnn_file

