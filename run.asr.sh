#!/bin/bash

# An example script for feature extraction of Homework 1

# Paths to different tools and data
eesen_path=$HOME/tools/eesen-offline-transcriber
audio_path=$HOME/hw1/audio
asr_output_path=$HOME/hw1/asrtxt

# Run ASR transcriber (speech2text)
echo "Cleaning ASR transcrition output"
rm -rf $eesen_path/build/output/*

echo "Running ASR transcriber (speech2text)"
for file in $(ls $audio_path/*)
do
    echo $eesen_path/speech2text.sh $file
    $eesen_path/speech2text.sh $file
done

echo "Copying ASR transcriptions (txt) to HW directory"
mkdir -p asrtxt
cp $eesen_path/build/output/*.txt $asr_output_path

# Now we generate the ASR-based features. This requires a vocabulary file to available beforehand. Each video is represented by
# a vector which has the same dimension as the size of the vocabulary. The elements of this vector are the number of occurrences 
# of the corresponding word. The vector is normalized to be like a probability. 
# You can of course explore other better ways, such as TF-IDF, of generating these features.
echo "[Fake] Creating ASR features"
mkdir -p asrfeat
python scripts/create_asrfeat.py vocab list/all.video #|| exit 1;

