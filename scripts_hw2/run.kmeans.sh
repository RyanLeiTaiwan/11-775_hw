#!/bin/bash

cluster_num=1000
video_list=../list_hw2/train_dev.video
feat_path=../sift
feat_suffix=".sift.pk"
model_path=../kmeans_models

# Now trains a k-means model using the sklearn package
echo "Training k-means models"
mkdir -p $model_path

#python2 scripts/train_kmeans.py select.mfcc.csv $cluster_num kmeans_models/kmeans.${cluster_num}.model || exit 1;
python2 train_kmeans.py $cluster_num $video_list $feat_path $feat_suffix $model_path || exit 1;

# Now that we have the k-means model, we can represent a whole video with the histogram of its MFCC vectors over the clusters. 
# Each video is represented by a single vector which has the same dimension as the number of clusters. 

#echo "Creating k-means cluster vectors"
#python2 scripts/create_kmeans.py kmeans_models/kmeans.${cluster_num}.model $cluster_num list/all.video || exit 1;


