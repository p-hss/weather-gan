#!/bin/bash

module load singularity
source /p/system/packages/spack/share/spack/setup-env.sh
spack load squashfs@4.4%gcc@8.3.0
mkdir -p /tmp/singularity/mnt/session

cd /home/hess/projects/weather-gan/

singularity run --nv --bind /p /home/hess/projects/container/singularity-pytorch/stack_v4.sif python main.py

exit
