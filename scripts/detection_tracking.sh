#!/bin/bash
set -o errexit

for cell_dir in "Data/Data-April20"/*
do
    echo $cell_dir
    #inference -->#comment image resizing in test if already 1024x1024
    python src/InferenceRetinaNet-Fake-Augmenations.py --fold Fold-ALL --dest_test $cell_dir
    #renumber images_resized from 0 to n instead of random numbering while keeping the correct numbering in images dir
    python src/correct_numbering.py --path  $cell_dir
    #node tracking
    python src/NodeTracking.py --dir $cell_dir --imgs images_resized/ --type .jpg
    #rename all dirs back to original numbering
    python src/correct_naming_after_tracking.py --path $cell_dir
done