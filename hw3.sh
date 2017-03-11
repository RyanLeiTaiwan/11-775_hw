# For each event, run the validation script of the "best" (highest MAP) feature
P001_pipeline=pipeline2
P002_pipeline=pipeline1
P003_pipeline=pipeline1
train_video_label=list_hw3/train_label

for feat in mfcc asr imtraj sift_uniform cnn_uniform
do  
    feat_path=$feat"_pred"
    # Append corresponding feature files to model_str
    feat_str="$feat_str $feat_path/train.$feat.pk"
done

echo "Running cross-validation script for event P001..."
P001_result=$(python2 "scripts_hw3/validate_"$P001_pipeline".py" P001 $train_video_label $feat_str)
echo "Running cross-validation script for event P002..."
P002_result=$(python2 "scripts_hw3/validate_"$P002_pipeline".py" P002 $train_video_label $feat_str)
echo "Running cross-validation script for event P003..."
P003_result=$(python2 "scripts_hw3/validate_"$P003_pipeline".py" P003 $train_video_label $feat_str)
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

