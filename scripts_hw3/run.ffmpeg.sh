#!/bin/bash

# ffmpeg for extracting video keyframes
# https://ffmpeg.org/ffmpeg.html

# Paths to different tools; 
ffmpeg_path=$HOME/tools/FFmpeg/build/bin/
export PATH=$ffmpeg_path:$PATH
export LD_LIBRARY_PATH=$ffmpeg_path/libs:$LD_LIBRARY_PATH
video_path=$HOME/video_local
video_list=../list_hw3/all.video
temp_path=../temp
keyframe_path=../keyframe_uniform

# Use ffmpeg to cut the videos for only the first $cut seconds (output mp4)
#cut=90
#width=640
#height=480
#input_fps=24
output_fps=0.5

mkdir -p $temp_path $keyframe_path

#for line in HVC1125 HVC51 HVC6170
for line in $(cat $video_list)
do
    # Use the original video resolution and length. Do NOT cut and scale AT ALL
    #ffmpeg -y -ss 0 -i $video_path/$line.mp4 -strict experimental -t $cut -r 15 -vf scale=$width"x"$height,setdar=4:3 $temp_path/tmp_$cut.mp4 || exit 1;
    mkdir -p $keyframe_path/$line
    # Sample frames at a constant FPS
    ffmpeg -i $video_path/$line.mp4 -q:v 2 -vf fps=$output_fps $keyframe_path/$line/%06d.jpg || exit 1;
    # https://gist.github.com/savvot/9e4316dc68f6111f7b1f
    # http://superuser.com/questions/669716/how-to-extract-all-key-frames-from-a-video-clip
    #ffmpeg -i $video_path/$line.mp4 -q:v 2 -vf select='eq(pict_type\,I)' -vsync vfr $keyframe_path/$line/%06d.jpg || exit 1;
    # TA's command
    #ffmpeg -i $video_path/$line.mp4 -vf select='eq(pict_type\,I)',setpts='N/(25*TB)' $keyframe_path/$line/%06d.jpg || exit 1;
    #ffmpeg -i $temp_path/tmp_$cut.mp4 -vf select='eq(pict_type\,I)',setpts='N/(25*TB)' $keyframe_path/$line/%06d.jpg || exit 1;
done

