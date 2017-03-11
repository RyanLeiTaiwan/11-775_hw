#!/bin/bash
# MED Pipeline 3: Two-level learning (5 SVMs + Logistic Regression)

# With labels
train_video_label=../list_hw3/train_label
# Without labels
test_video_list=../list_hw3/test_hw3
pipeline=pipeline3
pipeline_path=../$pipeline
# Output folders for HW3
score_results=$pipeline_path/score_results
classification_results=$pipeline_path/classification_results
mkdir -p $score_results
mkdir -p $classification_results

echo "############################"
echo "#      MED Pipeline 3      #"
echo "############################"
#for event in P001
for event in P001 P002 P003
do
    echo
    echo "=========  Event $event  ========="

    # [1] Train an ensemble classifier
    echo "##  TRAIN (ENSEMBLE)  ##"
    feat_model_str=""
    for feat in mfcc asr imtraj sift_uniform cnn_uniform
    do
        feat_path=../$feat"_pred"
        # Append corresponding feature files to model_str
        feat_model_str="$feat_model_str $feat_path/train.$feat.pk $feat_path/svm.$feat.$event.model"
    done
    #echo $feat_model_str
    python2 train_pipeline3.py $event $train_video_label $feat_model_str $pipeline_path/lr.$event.model

    # [2] Cross-validation (score reporting only)
    echo
    echo "##  VALIDATE  ##"
    python2 validate_pipeline3.py $event $train_video_label $feat_model_str

    # [3] Ensemble testing (generating "score results" and "classification results")
    echo
    echo "##  TEST  ##"
    feat_model_str=""
    for feat in mfcc asr imtraj sift_uniform cnn_uniform
    do
        feat_path=../$feat"_pred"
        # Append corresponding feature files to model_str
        feat_model_str="$feat_model_str $feat_path/test.$feat.pk $feat_path/svm.$feat.$event.model"
    done
    #echo $feat_model_str
    python2 test_pipeline3.py $event $feat_model_str $pipeline_path/lr.$event.model $score_results $classification_results
done

