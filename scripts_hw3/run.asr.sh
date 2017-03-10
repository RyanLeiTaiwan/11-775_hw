#!/bin/bash

# Paths to different tools and data
eesen_path=$HOME/tools/eesen-offline-transcriber
all_video_list=../list_hw3/all.video
train_video_list=../list_hw3/train_dev.video
test_video_list=../list_hw3/test_hw3
audio_path=$HOME/hw3/audio
feat=asr
asrtxt_path=../$feat
model_path=../$feat"_pred"
vocab_file=$model_path/vocab.txt
model_file=$model_path/$feat.tf.model

#mkdir -p $asrtxt_path
#echo "Running ASR transcriber (speech2text)"
#for file in HVC72
#for file in HVC5867 HVC5868
#for file in $(cat $all_video_list)
#do
    # It is very important to remove all previous files inside
    # EESEN build directory before start decoding ANY file
    #rm -rf $eesen_path/build/*
    # Run the ASR script
    #$eesen_path/speech2text.sh $audio_path/$file.wav
    # Copy the output txt to HW directory
    #cp $eesen_path/build/output/$file.txt $asrtxt_path
    # Create an empty ASR transcription if EESEN fails
    #touch $asrtxt_path/$file.txt
#done

# Now we generate the ASR-based features. This requires a vocabulary file to available beforehand.
# Each video is represented by a vector which has the same dimension as the size of the vocabulary.
# The elements of this vector are the number of occurrences of the corresponding word.
# The vector is normalized to be like a probability. 
# You can of course explore other better ways, such as TF-IDF, of generating these features.

echo "Creating ASR features"
mkdir -p $model_path
python2 train_wc.py $asrtxt_path $train_video_list $vocab_file $model_file
python2 test_wc.py $asrtxt_path $train_video_list $model_file $model_path/train.$feat.pk
python2 test_wc.py $asrtxt_path $test_video_list $model_file $model_path/test.$feat.pk

