#!/bin/bash
branch=$1
namefile=$2
tool=$3
quickcheck=$4
wait_or_run=${5:-1} # 1 is wait, 0 is run. Default is wait

# Check if there are changes, and count the number of change. It fails if the change in a submodule is zero. Also you need to add the numbers together and git diff the root dir.
 # DIR=$(pwd) & git submodule foreach 'git diff --numstat | grep -oE '^\s*[0-9]+'' >> ./tmp/pathd.txt

port=8888

#cd code/
#git submodule foreach 'git pull origin master --rebase -X theirs'
#git pull -X theirs --rebase origin --recurse-submodules 
#git checkout $1
#cd ..
# python working directory is where python gets called
echo Running docker on file $namefile

result=-1
first_time=1
# if the number of jupy docker is lower than the number of total docker 
# if jupydock=0 and tot docker=0=>false (you can run)
# if jupydock=1 and tot docker=1=>false (you can run)
# if jupydock=1 and totdock=2=>true (cannot run)
if [ $(docker ps -aq -f name=jupy | wc -l) -lt $(docker ps | wc -l) ] && [ $(docker ps | wc -l) -gt 1 ] && [ $wait_or_run -eq 1 ]
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

port=$(shuf -i 8000-8900 -n 1) 
echo port chosen is $port
		
#fi
echo ------------------ docker ready to run ------------------
docker run \
	--shm-size 2G \
	--name $(tmux display-message -p '#S') \
	-it \
	--gpus all \
	--ipc=host \
	--rm -p $port:8888 \
	-v  val:/work \
	-v ~/.git:/work/.git \
	-v ~/code:/work/code \
	-v ~/models:/work/models \
	-v ~/results:/work/results \
	-v ~/scripts:/work/scripts \
	-v ~/Unity-ML-Agents-Computer-Vision:/work/Unity-ML-Agents-Computer-Vision \
	-v ~/data:/work/data \
	-e CHOWN_HOME=yes \
	-e CHOWN_EXTRA_OPTS=’-R’ \
	--user root \
	-e DISPLAY=:1 \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	mmrl/val-pytorch /bin/bash -c "sh start_x_daemon.sh; $tool ./code/$namefile $quickcheck; exit" 


# port=$(shuf -i 8000-8900 -n 1) 
# docker run \
                # --shm-size 2G \
                # --name exp$(shuf -i 0-9999 -n 1) \
                # -it \
                # --gpus all \
                # --ipc=host \
                # --rm -p $port:8888 \
                # -v  val:/work \
                # -v ~/code:/work/code \
                # -v ~/models:/work/models \
                # -v ~/results:/work/results \
                # -v ~/scripts:/work/scripts \
                # -v ~/Unity-ML-Agents-Computer-Vision:/work/Unity-ML-Agents-Computer-Vision \
                # -v ~/data:/work/data \
                # -e CHOWN_HOME=yes \
                # -e CHOWN_EXTRA_OPTS=’-R’ \
                # --user root \
                # -e DISPLAY=:1 \
                # -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
                # --runtime=nvidia \
                # mmrl/val-pytorch /bin/bash -c "sh start_x_daemon.sh && $tool ./code/$namefile $quickcheck; exit"




