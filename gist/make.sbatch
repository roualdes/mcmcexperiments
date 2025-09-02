#!/bin/bash
#SBATCH -J dynamic-disBatch
#SBATCH --nodes=1
#SBATCH --time=1-0
#SBATCH --mail-type=ALL --mail-user=eroualdes@csuchico.edu

. env.sh

cd /mnt/home/eroualdes/ceph/bridgestan
make -j8 test_models

cd /mnt/home/eroualdes/ceph/cmdstan
make -j8 build
