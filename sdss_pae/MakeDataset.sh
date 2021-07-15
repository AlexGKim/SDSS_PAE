#!/bin/bash
#SBATCH -A m1759
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q regular
#SBATCH -t 7:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 10
#SBATCH --gpus-per-task=1
#SBATCH --output=stdout.out
#SBATCH --error=stderr.out

module load cgpu
export SLURM_CPU_BIND="cores"
module load python
source activate tf22

srun tfds build --data_dir=/global/cscratch1/sd/vboehm/Datasets/SDSS_BOSS_all --register_checksums

