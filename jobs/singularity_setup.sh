#!/bin/bash

module load singularity
source /p/system/packages/spack/share/spack/setup-env.sh
spack load squashfs@4.4%gcc@8.3.0
mkdir -p /tmp/singularity/mnt/session
