#!/bin/bash
branch=$1
namefile=$2
wait_or_run=$3 # 1 is wait, 0 is run. Default is wait
python_args=$4
port=8888
echo $python_args

cd code/
git pull -X theirs --rebase origin --recurse-submodules  
git checkout $1
# you shuld specify the branch at some point
cd ..
# python working directory is where python gets called
echo Running docker on file $namefile

result=-1
first_time=1
if [ $(docker ps | wc -l) -gt 1 ] && [ $wait_or_run -eq 1 ]
then
	while [ $(docker ps | wc -l) -gt 1 ]
	do
		if [ $first_time -eq 1 ]
		then
			echo Process is busy... we will wait
			first_time=0
		fi
		sleep 2
	done
fi

port=$(shuf -i 8889-8900 -n 1) 
echo port chosen is $port
		
#fi
echo ------------------ docker ready to run ------------------
docker run \
		--gpus all \
		--ipc=host \
		--rm -p $port:8888 \
		-v  val:/work \
		-v ~/code:/work/code \
		-v ~/models:/work/models \
		-v ~/results:/work/results \
		-v ~/scripts:/work/scripts \
		-v ~/data:/work/data \
		-e CHOWN_HOME=yes \
		-e CHOWN_EXTRA_OPTS=’-R’ \
		--user root \
		mmrl/dl-pytorch-nept python ./code/$namefile $python_args

