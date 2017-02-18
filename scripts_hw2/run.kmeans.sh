#!/bin/bash

cluster_num=1000
video_list=../list_hw2/train_dev.video
#video_list=../list_hw2/test_hw2
feat_path=../sift
feat_suffix=".sift.pk"
model_path=../kmeans_models
model_file=$model_path/sift.${cluster_num}.model
BoW_path=../sift_pred
BoW_file=$BoW_path/train.sift.pk

# Now trains a k-means model using the sklearn package
#echo "Training k-means models"
#mkdir -p $model_path

#python2 train_kmeans.py $cluster_num $video_list $feat_path $feat_suffix $model_file

# Now that we have the k-means model, we can represent a whole video with the histogram of its MFCC vectors over the clusters. 
# Each video is represented by a single vector which has the same dimension as the number of clusters. 

echo "Creating K-means BoW features"
mkdir -p $BoW_path
python2 test_kmeans.py $cluster_num $video_list $feat_path $feat_suffix $model_file $BoW_file


