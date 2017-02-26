#!/bin/bash

# Paths to different tools and data
eesen_path=$HOME/tools/eesen-offline-transcriber
video_list=../list_hw3/all.video
audio_path=$HOME/hw3/audio
asrtxt_path=../asr
asrfeat_path=../asr_pred

mkdir -p $asrtxt_path
echo "Running ASR transcriber (speech2text)"
#for file in HVC72
for file in $(cat $video_list)
do
    # It is very important to remove all previous files inside
    # EESEN build directory before start decoding ANY file
    rm -rf $eesen_path/build/*
    # Run the ASR script
    $eesen_path/speech2text.sh $audio_path/$file.wav
    # Copy the output txt to HW directory
    cp $eesen_path/build/output/$file.txt $asrtxt_path
done

# Now we generate the ASR-based features. This requires a vocabulary file to available beforehand.
# Each video is represented by a vector which has the same dimension as the size of the vocabulary.
# The elements of this vector are the number of occurrences of the corresponding word.
# The vector is normalized to be like a probability. 
# You can of course explore other better ways, such as TF-IDF, of generating these features.

#echo "Creating ASR features"
#mkdir -p $asrfeat_path
#python2 scripts/create_asrfeat.py $asrtxt_path list/all.video vocab $asrfeat_path || exit 1;

