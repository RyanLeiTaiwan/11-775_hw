#!/bin/bash
s3_video_path=$HOME/video
local_video_path=$HOME/video_local
mkdir -p $local_video_path

# Only download positive videos of each class
for class in P001 P002 P003
do
    echo "Downloading positive training videos for class" $class"..."
    mkdir -p $local_video_path/train_pos_$class
    for file in $(cat list_hw2/train_$class)
    do  
        echo $file".mp4"
        cp $s3_video_path/$file.mp4 $local_video_path/train_pos_$class
    done

    # Testing video list is not yet available
    #echo "Downloading positive testing videos for class" $class"..."
    #mkdir -p $local_video_path/test_pos_$class
    #for file in $(cat list/test | grep $class | awk '{print $1}')
    #do  
    #    echo $file".mp4"
    #    cp $s3_video_path/$file.mp4 $local_video_path/test_pos_$class
    #done
done

