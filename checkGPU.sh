#$ /bin/bash

gpu0_usage=$(nvidia-smi stats -d gpuUtil -c 1 | grep "^0," | head -n 1 | cut -f5 -d' ')
gpu1_usage=$(nvidia-smi stats -d gpuUtil -c 1 | grep "^1," | head -n 1 | cut -f5 -d' ')
gpu2_usage=$(nvidia-smi stats -d gpuUtil -c 1 | grep "^2," | head -n 1 | cut -f5 -d' ')
gpu3_usage=$(nvidia-smi stats -d gpuUtil -c 1 | grep "^3," | head -n 1 | cut -f5 -d' ')
gpu0_usage_check=`echo "$gpu0_usage < 5" | bc -l` 
gpu1_usage_check=`echo "$gpu1_usage < 5" | bc -l`
gpu2_usage_check=`echo "$gpu2_usage < 5" | bc -l` 
gpu3_usage_check=`echo "$gpu3_usage < 5" | bc -l`

echo $gpu0_usage_check
echo $gpu1_usage_check
echo $gpu2_usage_check
echo $gpu3_usage_check

gpuid=0
if [ "$gpu0_usage_check" -eq 1 ]; then
  gpuid=1
elif [ "$gpu1_usage_check" -eq 1 ]; then
  gpuid=2
elif [ "$gpu2_usage_check" -eq 1 ]; then
  gpuid=3
elif [ "$gpu3_usage_check" -eq 1 ]; then
  gpuid=4
else
  echo "There are no free GPUs on this machine"
  exit 1
fi
echo "GPU "$gpuid" will be used."