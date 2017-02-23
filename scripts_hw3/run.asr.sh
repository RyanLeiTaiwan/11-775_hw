#!/bin/bash

# An example script for feature extraction of Homework 1

# Paths to different tools and data
eesen_path=$HOME/tools/eesen-offline-transcriber
video_list=../list_hw3/all.video
audio_path=$HOME/hw3/audio
asrtxt_path=../asr
asrfeat_path=../asr_pred

# Run ASR transcriber (speech2text)
echo "Cleaning ASR transcrition output"
echo "rm -rf $eesen_path/build/output/*"
rm -rf $eesen_path/build/output/*
echo "Running ASR transcriber (speech2text)"
for file in $(cat $video_list)
do
    echo "$eesen_path/speech2text.sh $audio_path/$file.wav"
    $eesen_path/speech2text.sh $audio_path/$file.wav
done

echo "Copying ASR transcriptions (txt) to HW directory"
mkdir -p $asrtxt_path
echo "cp $eesen_path/build/output/*.txt $asrtxt_path"
cp $eesen_path/build/output/*.txt $asrtxt_path

# Now we generate the ASR-based features. This requires a vocabulary file to available beforehand.
# Each video is represented by a vector which has the same dimension as the size of the vocabulary.
# The elements of this vector are the number of occurrences of the corresponding word.
# The vector is normalized to be like a probability. 
# You can of course explore other better ways, such as TF-IDF, of generating these features.

#echo "Creating ASR features"
#mkdir -p $asrfeat_path
#python2 scripts/create_asrfeat.py $asrtxt_path list/all.video vocab $asrfeat_path || exit 1;

