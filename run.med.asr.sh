#!/bin/bash

# An example script for multimedia event detection (MED) of Homework 1
# Before running this script, you are supposed to have the features by running run.feature.sh 

# Note that this script gives you the very basic setup. Its configuration is by no means the optimal. 
# This is NOT the only solution by which you approach the problem. We highly encourage you to create
# your own setups. 

# Paths to different tools; 
map_path=/home/ubuntu/tools/mAP
#map_path=..//tools/mAP
export PATH=$map_path:$PATH

echo ""
echo "#####################################"
echo "#       MED with ASR Features       #"
echo "#####################################"
mkdir -p asr_pred
# iterate over the events
feat_dim_asr=4677  # This is calculated in scripts/create_asrfeat.py
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # Now train a svm model
  python2 scripts/train_svm.py $event "asrfeat" $feat_dim_asr asr_pred/svm.$event.model || exit 1;

  # Apply the svm model to *ALL* the testing videos;
  # Output the score of each testing video to a file ${event}_pred 
  python2 scripts/test_svm.py asr_pred/svm.$event.model "asrfeat" $feat_dim_asr asr_pred/${event}_pred || exit 1;

  # Compute the average precision by calling the mAP package
  ap list/${event}_test_label asr_pred/${event}_pred

  # Compute the class accuracy
  python2 scripts/class_accuracy.py list/${event}_test_label asr_pred/${event}_pred
done

