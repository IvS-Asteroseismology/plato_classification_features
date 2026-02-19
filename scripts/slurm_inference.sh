#!/bin/bash
#SBATCH --job-name=plato_fov_features
#SBATCH --account=ivsusers
#SBATCH --time=23:59:00
#SBATCH --output=/home/pabloh/stdout_plato.log
#SBATCH --error=/home/pabloh/stderr_plato.log
#SBATCH --cpus-per-task=64
#SBATCH --hint=multithread
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=250M
#SBATCH --partition=normal
#SBATCH --qos=normal

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export POLARS_MAX_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

repo=/STER/mamsie_tess/TIC/plato_classification_features
uv run $repo/src/runner.py inference $repo/results/light_curves/ $repo/results --n-jobs=$SLURM_CPUS_PER_TASK --backend=loky
