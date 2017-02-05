#!/bin/bash

# An example script for feature extraction of Homework 1

# Paths to different tools and data
eesen_path=$HOME/tools/eesen-offline-transcriber
audio_path=$HOME/hw1/audio

# Run ASR transcriber (speech2text)
echo "Cleaning ASR transcrition output"
echo "rm -rf $eesen_path/build/output/*"
rm -rf $eesen_path/build/output/*

echo "Running ASR transcriber (speech2text)"
# Break the all.video list into multiple parts and run on different VMs
for file in $(cat list/all.video.1)
do
    # echo "$eesen_path/speech2text.sh $audio_path/$file.wav"
    $eesen_path/speech2text.sh $audio_path/$file.wav
done

echo "Copying ASR transcriptions (txt) to HW directory"
mkdir -p asrtxt
echo "cp $eesen_path/build/output/*.txt asrtxt"
cp $eesen_path/build/output/*.txt asrtxt

# Now we generate the ASR-based features. This requires a vocabulary file to available beforehand. Each video is represented by
# a vector which has the same dimension as the size of the vocabulary. The elements of this vector are the number of occurrences 
# of the corresponding word. The vector is normalized to be like a probability. 
# You can of course explore other better ways, such as TF-IDF, of generating these features.
echo "[Fake] Creating ASR features"
mkdir -p asrfeat
python scripts/create_asrfeat.py vocab list/all.video || exit 1;

