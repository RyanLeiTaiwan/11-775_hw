# For each event, run the validation script of the "best" (highest MAP) feature
P001_feat=cnn
P002_feat=cnn
P003_feat=cnn
train_video_label=list_hw2/train_label

echo "Running cross-validation script for event P001..."
P001_result=$(python2 scripts_hw2/validate_svm.py P001 $train_video_label $P001_feat"_pred"/train.$P001_feat.pk $P001_feat"_pred"/svm.P001.model)
echo "Running cross-validation script for event P002..."
P002_result=$(python2 scripts_hw2/validate_svm.py P002 $train_video_label $P002_feat"_pred"/train.$P002_feat.pk $P002_feat"_pred"/svm.P002.model)
echo "Running cross-validation script for event P003..."
P003_result=$(python2 scripts_hw2/validate_svm.py P003 $train_video_label $P003_feat"_pred"/train.$P003_feat.pk $P003_feat"_pred"/svm.P003.model)
echo

#echo $P001_result
#echo $P002_result
#echo $P003_result

# Actual reporting
echo 3 FOLD CROSS VALIDATION RESULT \(MAP\)

echo P001: $(echo $P001_result | awk '{print $2}')
echo P002: $(echo $P002_result | awk '{print $2}')
echo P003: $(echo $P003_result | awk '{print $2}')

echo 3 FOLD CROSS VALIDATION RESULT \(CLASS ACCURACY\)

echo P001: $(echo $P001_result | awk '{print $4}')
echo P002: $(echo $P002_result | awk '{print $4}')
echo P003: $(echo $P003_result | awk '{print $4}')

echo 3 FOLD CROSS VALIDATION RESULT \(TRUE POSITIVE RATE\)

echo P001: $(echo $P001_result | awk '{print $6}')
echo P002: $(echo $P002_result | awk '{print $6}')
echo P003: $(echo $P003_result | awk '{print $6}')

echo 3 FOLD CROSS VALIDATION RESULT \(TRUE NEGATIVE RATE\)

echo P001: $(echo $P001_result | awk '{print $8}')
echo P002: $(echo $P002_result | awk '{print $8}')
echo P003: $(echo $P003_result | awk '{print $8}')

