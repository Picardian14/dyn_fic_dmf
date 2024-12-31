#!/bin/bash
#SBATCH --job-name=cnt_fit_bayesian_static_fc
#SBATCH --cpus-per-task=14
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --output=outputs/g_lr_dyn_%A_%a.out
#SBATCH --error=outputs/g_lr_dyn%A_%a.err
#SBATCH --chdir=/network/iss/home/ivan.mindlin/
# Load Singularity module
module load singularity

# Define paths
SINGULARITY_IMAGE=/network/iss/home/ivan.mindlin/ubuntu_focal_conda.sif

# Run Python script inside Singularity container
singularity exec $SINGULARITY_IMAGE bash -c 'source /opt/anaconda/3/2023.07-2/base/etc/profile.d/conda.sh && conda activate fic && cd dyn_fic_dmf/python && python3 cnt_fit_bayesian_static_fc.py ' > ./g_lr.out 