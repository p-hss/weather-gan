#!/bin/bash

#SBATCH --qos=medium
##SBATCH --qos=short
##SBATCH --qos=long
#SBATCH --job-name=weather-gan
#SBATCH --account=tipes
#SBATCH --output=out/%x-%j.out
#SBATCH --error=out/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
##SBATCH --mem=32GB
#SBATCH --mem=64GB
#SBATCH --ntasks=1
##SBATCH --cpus-per-task=8
#SBATCH --cpus-per-task=16
##SBATCH --time=06-23:59:00

module load singularity
source /p/system/packages/spack/share/spack/setup-env.sh
spack load squashfs@4.4%gcc@8.3.0
mkdir -p /tmp/singularity/mnt/session

cd /home/hess/projects/weather-gan/

singularity run --nv --bind /p /home/hess/projects/container/singularity-pytorch/stack_v4.sif python main.py
