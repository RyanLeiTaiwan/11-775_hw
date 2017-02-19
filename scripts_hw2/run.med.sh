#!/bin/bash

# Paths to different tools; 
map_path=../../tools/mAP
export PATH=$map_path:$PATH

# With labels
train_video_list=../list_hw2/train_dev.video
train_video_label=../list_hw2/train_label
# Without labels
test_video_list=../list_hw2/test_hw2
# Output folders for HW2
score_results=../score_results
classification_results=../classification_results
mkdir -p $score_results
mkdir -p $classification_results

#for feat in imtraj
for feat in imtraj sift cnn
do
    echo
    echo "#####################################"
    featUpper=$(echo $feat | tr [:lower:] [:upper:])
    echo "#      MED with $featUpper Features"
    echo "#####################################"
    #for event in P001
    for event in P001 P002 P003
    do
        echo
        echo "=========  Event $event  ========="
        # feature path == model path for convenience
        feat_path=../$feat"_pred"

        # [1] SVM training
        echo "##  TRAIN  ##"
        python2 train_svm.py $event $train_video_label $feat_path/train.$feat.pk $feat_path/svm.$event.model

        # [2] SVM cross-validation (score reporting only)
        echo
        echo "##  VALIDATE  ##"
        python2 validate_svm.py $event $train_video_label $feat_path/train.$feat.pk $feat_path/svm.$event.model

    done
done

# [3] SVM testing (generating "score results" and "classification results")
# Generate results ONLY for the "best" feature of each event
echo
echo "##  TEST  ##"
P001_feat=cnn
P002_feat=cnn
P003_feat=cnn
echo "=========  Event P001  ========="
python2 test_svm.py P001 ../$P001_feat"_pred"/test.$P001_feat.pk ../$P001_feat"_pred"/svm.P001.model $score_results $classification_results
echo "=========  Event P002  ========="
python2 test_svm.py P002 ../$P002_feat"_pred"/test.$P002_feat.pk ../$P002_feat"_pred"/svm.P002.model $score_results $classification_results
echo "=========  Event P003  ========="
python2 test_svm.py P003 ../$P003_feat"_pred"/test.$P003_feat.pk ../$P003_feat"_pred"/svm.P003.model $score_results $classification_results

