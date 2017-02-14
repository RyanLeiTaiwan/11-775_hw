#!/bin/bash
list_file=train_dev

for class in P001 P002 P003 NULL
do
    echo "Splitting class" $class from list file $list_file

    cat $list_file | grep $class | awk '{print $1}' > train_$class
done

