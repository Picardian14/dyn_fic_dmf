#!/bin/bash
#SBATCH --job-name=homeostatic_grid688
#SBATCH --cpus-per-task=24
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --output=outputs/homeostatic688_%A_%a.out
#SBATCH --error=outputs/homeostatic688%A_%a.err
#SBATCH --chdir=/network/lustre/iss02/home/ivan.mindlin/
# Load Singularity module
module load singularity

# Define paths
SINGULARITY_IMAGE=/network/lustre/iss02/home/ivan.mindlin/ubuntu_focal_conda.sif

# Run Python script inside Singularity container
singularity exec $SINGULARITY_IMAGE bash -c 'source /opt/anaconda/3/2023.07-2/base/etc/profile.d/conda.sh && conda activate fic && cd dyn_fic_dmf/python && python3 Homeostatic_Grid_DECAYvsLR6_88.py ' > ./hom_grid.out 