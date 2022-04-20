#!/bin/bash
project_to_build_path=Unity-ML-Agents-Computer-Vision
path=$(dirname $BASH_SOURCE)
call()
{
	args="./code/generate_datasets/dataset_utils/unity_builders/build_unity_scene_sequence.py -os "$os" -sc "$sc" -nsu "$nsu""
	# output_path_relative_to_scene="Builds/Builds"$os"/"$nsu"/k"$k"_nSt"$nSt"_nSc"$nSc"_nFt"$nFt"_nFc"$nFc"_sc"$sc"_g"$g""
	# echo trying to ssh...
	# ssh mynet "mkdir -p ~/"$project_to_build_path"/"$output_path_relative_to_scene""
	}

nsu=SequenceLearning
sc=128


os=win
nsu=SequenceLearning
# cd .. && call && python $args -k 16 -nSt 1 -nSc 0 -nFt 1 -nFc 0 -g 0 && cd scripts
# cd .. && call && python $args -k 1 -nSt 1 -nSc 1 -nFt 1 -nFc 1 -g 0 && cd scripts
# cd .. && call && python $args -k 32 -nSt 1 -nSc 1 -nFt 1 -nFc 1 -g 0 && cd scripts


os=linux
nsu=SequenceLearning
cd .. && call && python $args -k 16  -nSt 1 -nSc 0 -nFt 1 -nFc 0 -g 0 && cd scripts
# cd .. && call && python $args -k 32  -nSt 1 -nSc 1 -nFt 1 -nFc 1 -g 0 && cd scripts
# cd .. && call && python $args -k 1  -nSt 1 -nSc 1 -nFt 1 -nFc 1 -g 0 && cd scripts

echo "rsyncing /Builds_linux/, please wait..."
rsync -r -v -c --progress "./"$path"/../"$project_to_build_path"/Builds/Builds_linux/" mynet:~/"$project_to_build_path"/Builds/Builds_linux
ssh mynet "find ./"$project_to_build_path"/Builds/Builds_linux/* -name "*scene.*" -exec chmod -R 755 {} \;"

