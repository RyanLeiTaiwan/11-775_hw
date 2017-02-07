#!/bin/bash

# An example script for feature extraction of Homework 1

video_path=/home/ubuntu/video   # path to the directory containing all the videos.

# In this part, we train a clustering model to cluster the MFCC vectors. In order to speed up the clustering process, we
# select a small portion of the MFCC vectors. In the following example, we only select 20% randomly from each video. 
#echo "Pooling MFCCs (optional)"
#python scripts/select_frames.py list/train.video 0.2 select.mfcc.csv || exit 1;

# Now trains a k-means model using the sklearn package
echo "Training k-means models"
mkdir -p kmeans_models
# No longer specify number of clusters here (done in Python)
#python2 scripts/train_kmeans.py select.mfcc.csv $cluster_num kmeans_models/kmeans.${cluster_num}.model || exit 1;
python2 scripts/train_kmeans.py select.mfcc.csv kmeans_models || exit 1;

# Now that we have the k-means model, we can represent a whole video with the histogram of its MFCC vectors over the clusters. 
# Each video is represented by a single vector which has the same dimension as the number of clusters. 

# The optimal cluster_num determined by inertia
cluster_num=150
echo "Creating k-means cluster vectors"
python2 scripts/create_kmeans.py kmeans_models/kmeans.${cluster_num}.model $cluster_num list/all.video || exit 1;

# Now you can see that you get the bag-of-word representations under kmeans/. Each video is now represented
# by a {cluster_num}-dimensional vector.

