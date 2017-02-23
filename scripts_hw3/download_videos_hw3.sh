#!/bin/bash
video_list=../list_hw3/all.video
s3_video_path=$HOME/video
local_video_path=$HOME/video_local
mkdir -p $local_video_path

# Download EVERY video since we have 300GB storage
for video in $(cat $video_list)
do
    echo Downloading $video."mp4"
    cp $s3_video_path/$video.mp4 $local_video_path
done

