#$ -S /bin/bash
#$ -l hostname=cl9lx
#$ -cwd
#$ -V
#$ -o /nethome/trenslow/thesis/logs/
#$ -e /nethome/trenslow/thesis/logs/
export LD_LIBRARY_PATH="/nethome/trenslow/cuDNNv6/lib64:/usr/local/cuda-8.0/lib64/:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
python maxEntMulti.py bul rus
