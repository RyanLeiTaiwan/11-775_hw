#!/bin/bash
s3_video_path=$HOME/video
local_video_path=$HOME/video_local
mkdir -p $local_video_path

# Download positive training videos of each class
for class in P001 P002 P003
do
    echo "Downloading positive training videos for class" $class"..."
    mkdir -p $local_video_path/train_pos_$class
    for file in $(cat list_hw2/train_$class)
    do  
        echo $file".mp4"
        cp $s3_video_path/$file.mp4 $local_video_path/train_pos_$class
    done

done

# Wait until the test set is released
#echo "Downloading ALL test videos..."
#mkdir -p $local_video_path/test
#for file in $(cat list_hw2/test_hw2)
#do  
#    echo $file".mp4"
#    cp $s3_video_path/$file.mp4 $local_video_path/test
#done

