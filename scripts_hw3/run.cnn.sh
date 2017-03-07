#!/bin/bash

# Extract Cafee CNN features for each keyframe
PATH=/home/ubuntu/tools/caffe/python:$PATH

all_video_list=../list_hw3/all.video
train_video_list=../list_hw3/train_dev.video
keyframe_path=../keyframe_vfr
# Path to output CNN feature for individual videos
cnn_individual_path=../cnn_vfr
# Path to output the combined CNN feature file
cnn_combine_path=../cnn_pred
cnn_combine_train_file=$cnn_combine_path/train.cnn.uniform.pk
cnn_combine_test_file=$cnn_combine_path/test.cnn.uniform.pk

mkdir -p $cnn_individual_path $cnn_combine_path
python2 extract_cnn.py $all_video_list $keyframe_path $cnn_individual_path

# Combine individual pk's into an overall pk
#python2 combine_cnn.py $train_video_list $cnn_individual_path $cnn_combine_train_file
#python2 combine_cnn.py $test_video_list $cnn_individual_path $cnn_combine_test_file
