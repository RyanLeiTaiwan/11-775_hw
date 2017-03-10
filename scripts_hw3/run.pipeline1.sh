#!/bin/bash
# MED Pipeline 1: Linear combination of prediction scores

# With labels
train_video_label=../list_hw3/train_label
# Without labels
test_video_list=../list_hw3/test_hw3
pipeline=pipeline1
pipeline_path=../$pipeline
# Output folders for HW3
score_results=$pipeline_path/score_results
classification_results=$pipeline_path/classification_results
mkdir -p $score_results
mkdir -p $classification_results

echo "############################"
echo "#      MED Pipeline 1      #"
echo "############################"
#for event in P001
for event in P001 P002 P003
do
    echo
    echo "=========  Event $event  ========="

    # [1] DO NOT train an ensemble classifier
    echo "##  NO TRAIN  ##"

    # [2] Cross-validation (score reporting only) using 5 SVM models
    echo
    echo "##  VALIDATE (ENSEMBLE)  ##"
    feat_str=""
    for feat in mfcc asr imtraj sift_uniform cnn_uniform
    do
        feat_path=../$feat"_pred"
        # Append corresponding feature files to model_str
        feat_str="$feat_str $feat_path/train.$feat.pk"
    done
    #echo $feat_str
    #python2 validate_pipeline1.py $event $train_video_label $feat_str

    # [3] Ensemble testing (generating "score results" and "classification results")
    echo
    echo "##  TEST  ##"
    feat_model_str=""
    for feat in mfcc asr imtraj sift_uniform cnn_uniform
    do
        feat_path=../$feat"_pred"
        # Append corresponding feature and model files to feat_model_str
        feat_model_str="$feat_model_str $feat_path/test.$feat.pk $feat_path/svm.$feat.$event.model"
    done
    #echo $feat_model_str
    python2 test_pipeline1.py $event $feat_model_str $score_results $classification_results
done

