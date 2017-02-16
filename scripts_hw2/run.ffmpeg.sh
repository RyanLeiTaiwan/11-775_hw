#!/bin/bash

# Homework 2 ffmpeg video processing

# Paths to different tools; 
opensmile_path=$HOME/tools/opensmile-2.3.0/bin/linux_x64_standalone_static
ffmpeg_path=$HOME/tools/FFmpeg/build/bin/
export PATH=$opensmile_path:$ffmpeg_path:$PATH
export LD_LIBRARY_PATH=$ffmpeg_path/libs:$opensmile_path/lib:$LD_LIBRARY_PATH
video_path=$HOME/video   # path to the directory containing all the videos.

# Case 1: use ffmpeg to cut the videos for only the first $cut seconds (output mp4)
cut=600
width=640
height=480

mkdir -p temp keyframe_$cut

#for line in HVC51 
#for line in $(cat list_hw2/train_dev.video)
for line in $(cat list_hw2/test_hw2)
do
    ffmpeg -y -ss 0 -i $video_path/$line.mp4 -strict experimental -t $cut -r 15 -vf scale=$width"x"$height,setdar=4:3 temp/tmp_$cut.mp4
    mkdir -p keyframe_$cut/$line
    ffmpeg -i temp/tmp_$cut.mp4 -vf select='eq(pict_type\,I)',setpts='N/(25*TB)' keyframe_$cut/$line/%09d.jpg
done

