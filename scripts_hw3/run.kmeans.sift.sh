#!/bin/bash

cluster_num=1000
train_video_list=../list_hw3/train_dev.video
test_video_list=../list_hw3/test_hw3
feat=sift_uniform
feat_path=../$feat
feat_suffix=".sift.pk"
model_path=../kmeans_models
model_file=$model_path/$feat.${cluster_num}.model
BoW_path=../$feat"_pred"
train_BoW_file=$BoW_path/train.$feat.pk
test_BoW_file=$BoW_path/test.$feat.pk

# Now trains a k-means model using the sklearn package
echo "Training k-means models"
mkdir -p $model_path

python2 train_kmeans.py $cluster_num $train_video_list $feat_path $feat_suffix $model_file

# Now that we have the k-means model, we can represent a whole video with the histogram of its MFCC vectors over the clusters. 
# Each video is represented by a single vector which has the same dimension as the number of clusters. 

mkdir -p $BoW_path
echo "Creating training K-means BoW features"
python2 test_kmeans.py $cluster_num $train_video_list $feat_path $feat_suffix $model_file $train_BoW_file
echo "Creating testing K-means BoW features"
python2 test_kmeans.py $cluster_num $test_video_list $feat_path $feat_suffix $model_file $test_BoW_file


