#!/bin/bash

# Run PCA to reduce dimension of imtraj feature

video_list=../list_hw2/train_dev.video
feat_dir=../imtraj_local
feat_dim=32768
pca_output=../imtraj_pred/imtraj_pca.pk

mkdir -p ../imtraj_pred
python2 pca.py $video_list $feat_dir $feat_dim $pca_output || exit 1;

