#!/bin/bash
#SBATCH --job-name=homeostatic_grid50
#SBATCH --cpus-per-task=24
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --output=outputs/homeostatic50_%A_%a.out
#SBATCH --error=outputs/homeostatic50%A_%a.err
#SBATCH --chdir=/network/iss/home/ivan.mindlin/
# Load Singularity module
module load singularity

# Define paths
SINGULARITY_IMAGE=/network/iss/home/ivan.mindlin/ubuntu_focal_conda.sif

# Run Python script inside Singularity container
singularity exec $SINGULARITY_IMAGE bash -c 'source /opt/anaconda/3/2023.07-2/base/etc/profile.d/conda.sh && conda activate fic && cd dyn_fic_dmf/python && python3 Homeostatic_Grid_DECAYvsLR50.py ' > ./hom_grid.out 