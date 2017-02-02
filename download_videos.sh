#!/bin/bash
s3_video_path=$HOME/video
local_video_path=$HOME/video_local
mkdir -p $local_video_path

# Only download positive training videos of each class (10 * 3 = 30 videos)
for class in P001 P002 P003
do
    echo "Downloading positive training videos for class" $class"..."
    mkdir -p $local_video_path/$class
    for file in $(cat list/$class"_train" | grep $class | awk '{print $1}')
    do  
        echo $file".mp4"
        cp $s3_video_path/$file.mp4 $local_video_path/$class
    done
done

