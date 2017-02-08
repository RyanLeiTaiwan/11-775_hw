#!/bin/bash

map_path=../tools/mAP
export PATH=$map_path:$PATH

# Run the testing scripts

echo "================    MFCC    ================"
feat_dim_mfcc=150
python2 scripts/test_svm.py mfcc_pred/svm.P001.model "kmeans" $feat_dim_mfcc mfcc_pred/P001_pred
python2 scripts/test_svm.py mfcc_pred/svm.P002.model "kmeans" $feat_dim_mfcc mfcc_pred/P002_pred
python2 scripts/test_svm.py mfcc_pred/svm.P003.model "kmeans" $feat_dim_mfcc mfcc_pred/P003_pred

# Compute (M)AP
MAPP001=$(ap list/P001_test_label mfcc_pred/P001_pred | awk '{print $2}')
MAPP002=$(ap list/P002_test_label mfcc_pred/P002_pred | awk '{print $2}')
MAPP003=$(ap list/P003_test_label mfcc_pred/P003_pred | awk '{print $2}')
echo P001 MAP: $MAPP001
echo P002 MAP: $MAPP002
echo P003 MAP: $MAPP003

# Compute accuracy
CAP001=$(python2 scripts/class_accuracy.py list/P001_test_label mfcc_pred/P001_pred | awk '{print $3}')
CAP002=$(python2 scripts/class_accuracy.py list/P002_test_label mfcc_pred/P002_pred | awk '{print $3}')
CAP003=$(python2 scripts/class_accuracy.py list/P003_test_label mfcc_pred/P003_pred | awk '{print $3}')
echo P001 CLASS ACCURACY: $CAP001
echo P002 CLASS ACCURACY: $CAP002
echo P003 CLASS ACCURACY: $CAP003

echo "================    ASR    ================"
feat_dim_asr=4677
python2 scripts/test_svm.py asr_pred/svm.P001.model "asrfeat" $feat_dim_asr asr_pred/P001_pred
python2 scripts/test_svm.py asr_pred/svm.P002.model "asrfeat" $feat_dim_asr asr_pred/P002_pred
python2 scripts/test_svm.py asr_pred/svm.P003.model "asrfeat" $feat_dim_asr asr_pred/P003_pred

# Compute (M)AP
MAPP001=$(ap list/P001_test_label asr_pred/P001_pred | awk '{print $2}')
MAPP002=$(ap list/P002_test_label asr_pred/P002_pred | awk '{print $2}')
MAPP003=$(ap list/P003_test_label asr_pred/P003_pred | awk '{print $2}')
echo P001 MAP: $MAPP001
echo P002 MAP: $MAPP002
echo P003 MAP: $MAPP003

# Compute accuracy
CAP001=$(python2 scripts/class_accuracy.py list/P001_test_label asr_pred/P001_pred | awk '{print $3}')
CAP002=$(python2 scripts/class_accuracy.py list/P002_test_label asr_pred/P002_pred | awk '{print $3}')
CAP003=$(python2 scripts/class_accuracy.py list/P003_test_label asr_pred/P003_pred | awk '{print $3}')
echo P001 CLASS ACCURACY: $CAP001
echo P002 CLASS ACCURACY: $CAP002
echo P003 CLASS ACCURACY: $CAP003

