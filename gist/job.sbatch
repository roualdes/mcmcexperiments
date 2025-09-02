#!/bin/bash
#SBATCH -J dynamic-disBatch
#SBATCH --nodes=1
#SBATCH --time=1-0
#SBATCH --mail-type=ALL --mail-user=eroualdes@csuchico.edu

. env.sh
script=$(realpath run_disbatcher.py)

workdir=job$SLURM_JOBID
mkdir -p $workdir
cd $workdir

python $script
