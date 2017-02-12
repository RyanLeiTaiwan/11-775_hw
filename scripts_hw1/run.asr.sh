#!/bin/bash

# An example script for feature extraction of Homework 1

# Paths to different tools and data
eesen_path=$HOME/tools/eesen-offline-transcriber
audio_path=$HOME/hw1/audio
asrtxt_path=asrtxt
asrfeat_path=asrfeat

# Run ASR transcriber (speech2text)
echo "Cleaning ASR transcrition output"
echo "rm -rf $eesen_path/build/output/*"
rm -rf $eesen_path/build/output/*
echo "Running ASR transcriber (speech2text)"
# If necessary, break the all.video list into multiple parts and run on different VMs
for file in $(cat list/all.video)
#for file in $(cat list/all.video.1)
do
    echo "$eesen_path/speech2text.sh $audio_path/$file.wav"
    $eesen_path/speech2text.sh $audio_path/$file.wav
done

echo "Copying ASR transcriptions (txt) to HW directory"
mkdir -p asrtxt
echo "cp $eesen_path/build/output/*.txt $asrtxt_path"
cp $eesen_path/build/output/*.txt $asrtxt_path

# Now we generate the ASR-based features. This requires a vocabulary file to available beforehand.
# Each video is represented by a vector which has the same dimension as the size of the vocabulary.
# The elements of this vector are the number of occurrences of the corresponding word.
# The vector is normalized to be like a probability. 
# You can of course explore other better ways, such as TF-IDF, of generating these features.
echo "Creating ASR features"
mkdir -p $asrfeat_path
python2 scripts/create_asrfeat.py $asrtxt_path list/all.video vocab $asrfeat_path || exit 1;

