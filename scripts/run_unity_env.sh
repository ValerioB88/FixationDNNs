#!/bin/sh
dd=$(date +"%d%m%y_%H%M")
#folder=${1:-$dd}
folder=${1}${dd}

pip install -e ./code/external/ml-agents/ml-agents-envs; pip install -e ./code/external/ml-agents/ml-agents;
mkdir tmp; chmod -R 775 ./unity_project/Naturally-Acquired-Invariances/Build_linux/VisFoodValerioLeek/scene.x86_64 &&
DISPLAY=:0
mlagents-learn ./unity_project/VisualFoodCollector.yaml \
	--env=./unity_project/Naturally-Acquired-Invariances/Build_linux/VisFoodValerioLeek/scene.x86_64 \
	--run-id=../models/RL_unity_invariance/$folder --force --connection_cost linear
