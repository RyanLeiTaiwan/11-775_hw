#!/bin/bash

# An example script for feature extraction of Homework 1

# Paths to different tools; 
opensmile_path=/home/ubuntu/tools/opensmile-2.3.0/bin/linux_x64_standalone_static
ffmpeg_path=/home/ubuntu/tools/FFmpeg/build/bin/
export PATH=$opensmile_path:$ffmpeg_path:$PATH
export LD_LIBRARY_PATH=$ffmpeg_path/libs:$opensmile_path/lib:$LD_LIBRARY_PATH

# Two additional variables
video_path=/home/ubuntu/video   # path to the directory containing all the videos.
cluster_num=50        # the number of clusters in k-means. Note that 50 is by no means the optimal solution.
                      # You need to explore the best config by yourself.

# In this part, we train a clustering model to cluster the MFCC vectors. In order to speed up the clustering process, we
# select a small portion of the MFCC vectors. In the following example, we only select 20% randomly from each video. 
echo "Pooling MFCCs (optional)"
python scripts/select_frames.py list/train.video 0.2 select.mfcc.csv || exit 1;

# now trains a k-means model using the sklearn package
echo "Training the k-means model"
python scripts/train_kmeans.py select.mfcc.csv $cluster_num kmeans.${cluster_num}.model || exit 1;

# Now that we have the k-means model, we can represent a whole video with the histogram of its MFCC vectors over the clusters. 
# Each video is represented by a single vector which has the same dimension as the number of clusters. 
echo "Creating k-means cluster vectors"
python scripts/create_kmeans.py kmeans.${cluster_num}.model $cluster_num list/all.video || exit 1;

# Now you can see that you get the bag-of-word representations under kmeans/. Each video is now represented
# by a {cluster_num}-dimensional vector.

# Now we generate the ASR-based features. This requires a vocabulary file to available beforehand. Each video is represented by
# a vector which has the same dimension as the size of the vocabulary. The elements of this vector are the number of occurrences 
# of the corresponding word. The vector is normalized to be like a probability. 
# You can of course explore other better ways, such as TF-IDF, of generating these features.
echo "Creating ASR features"
mkdir -p asrfeat
python scripts/create_asrfeat.py vocab list/all.video || exit 1;

