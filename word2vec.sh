#$ -S /bin/bash
#$ -l hostname=cl10lx
#$ -cwd
#$ -V
#$ -o /nethome/trenslow/thesis/logs/
#$ -e /nethome/trenslow/thesis/logs/
python word2vec.py rus bul