#!/bin/bash

call()
{
	args="./code/generate_datasets/dataset_utils/unity_builders/build_unity_scene_generator.py -os "$os" -sc "$sc" -nsu "$nsu""
}



####### META LEARNING APPROACH (NO SEQUENCES)
os=win
sc=64
nsu=MetaLearning
cd ../

os=win
call &&  python $args -n 2 -k 3 -q 1
exit 
os=linux
call &&  python $args -n 1 -k 10 -q 1


exit
os=linux
call &&  python $args -n 1 -k 10 -q 1
exit
call &&  python $args -n 1 -k 15 -q 1
call &&  python $args -n 1 -k 15 -q 5
os=win
exit 

3.3 -1.7 1.3
