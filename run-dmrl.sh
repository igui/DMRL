#!/usr/bin/bash -l 

# Exit on errors :)
set -e

cd ~/DMRL

echo "Exporting variables"
export LD_LIBRARY_PATH="${VSC_DATA}/cuda-10.0/lib64:${LD_LIBRARY_PATH}"
export PATH="${VSC_DATA}/cuda-10.0/bin:${PATH}"

echo "Activating conda environment"
conda activate dmrl 

echo "Checking for GPU in Tensorflow"
echo 'import tensorflow as tf; print(tf.test.is_gpu_available())' | python

# echo "Running nvidia-smi monitor"
# nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used  --format=csv -l 5 &

# echo "Running mpstat for CPU monitoring"
# mpstat 5 &

echo "Executing!"
echo "Extra Parameters are $@"
python DMRL.py --dataset=Clothing-Custom $@

echo "Done!"
