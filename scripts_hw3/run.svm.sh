#!/bin/bash
# train/validate/test a default SVM classifier for each feature

train_video_label=../list_hw3/train_label

#for feat in asr imtraj
for feat in mfcc asr imtraj sift_uniform cnn_uniform 
do
    echo
    echo "====  Feature $feat  ===="
    #for event in P001
    for event in P001 P002 P003
    do
        echo "========  Event $event  ========"
        # feature path == model path for convenience
        feat_path=../$feat"_pred"
        echo "##  train SVM  ##"
        python2 train_svm.py $event $train_video_label $feat_path/train.$feat.pk $feat_path/svm.$feat.$event.model

        # Report SVM cross-validation performance for diagnosis ONLY
        echo "##  validate SVM  ##"
        python2 validate_svm.py $event $train_video_label $feat_path/train.$feat.pk $feat_path/svm.$feat.$event.model

        # Test SVM to output score_results and classification_results for diagnosis
        echo "##  test SVM  ##"
        python2 test_svm.py $event $feat_path/test.$feat.pk $feat_path/svm.$feat.$event.model $feat_path $feat_path
    done

done

