#!/bin/bash

# Run PCA to reduce dimension of imtraj feature

train_video_list=../list_hw3/train_dev.video
test_video_list=../list_hw3/test_hw3
feat=imtraj
feat_dir=../$feat"_local"
feat_dim=32768
pca_path=../$feat"_pred"
train_pca_file=$pca_path/train.$feat.pk
test_pca_file=$pca_path/test.$feat.pk
# Dimensionality is NOT known until after training
pca_dim=713
pca_model=$pca_path/pca.imtraj.${pca_dim}.model

mkdir -p $pca_path
echo "Training PCA for training data dimensionality reduction"
python2 pca.py train $train_video_list $feat_dir $feat_dim $train_pca_file $pca_model

echo
echo "Using existing PCA model for test data dimensionality reduction"
python2 pca.py test $test_video_list $feat_dir $feat_dim $test_pca_file $pca_model

