#!/bin/bash

# Paths to different tools; 
opensmile_path=$HOME/tools/opensmile-2.3.0/bin/linux_x64_standalone_static
ffmpeg_path=$HOME/tools/FFmpeg/build/bin/
export PATH=$opensmile_path:$ffmpeg_path:$PATH
export LD_LIBRARY_PATH=$ffmpeg_path/libs:$opensmile_path/lib:$LD_LIBRARY_PATH

video_list=../list_hw3/all.video
video_path=$HOME/video   # path to the directory containing all the videos.
mfcc_conf=../config/MFCC12_0_D_A.conf
temp_path=../temp
audio_path=../audio
mfcc_path=../mfcc
mkdir -p $temp_path $audio_path $mfcc_path

# This part does feature extraction, it may take quite a while if you have a lot of videos. Totally 3 steps are taken:
# 1. ffmpeg extracts the audio track from each video file into a wav file
# 2. The wav file may contain 2 channels. We always extract the 1st channel using sox (instead of ch_wave)
# 3. SMILExtract generates the MFCC features for each wav file
#    The config file MFCC12_0_D_A.conf generates 13-dim MFCCs at each frame, together with the 1st and 2nd deltas. So you 
#    will see each frame totally has 39 dims. 
#    Refer to Section 2.5 of this document http://web.stanford.edu/class/cs224s/hw/openSMILE_manual.pdf for better configuration
#    (e.g., normalization) and other feature types (e.g., PLPs )     
#for line in HVC51
for line in $(cat $video_list)
do
    ffmpeg -y -i $video_path/${line}.mp4 -f wav $temp_path/tmp.wav
    #ffmpeg -y -i $video_path/${line}.mp4 -ac 1 -f wav audio/$line.wav
    sox $temp_path/tmp.wav -c 1  $audio_path/$line.wav
    SMILExtract -C $mfcc_conf -I $audio_path/$line.wav -O $mfcc_path/$line.mfcc.csv
done
# You may find the number of MFCC files mfcc/*.mfcc.csv is slightly less than the number of the videos. This is because some of the videos
# don't hae the audio track. For example, HVC1221, HVC1222, HVC1261, HVC1794 

