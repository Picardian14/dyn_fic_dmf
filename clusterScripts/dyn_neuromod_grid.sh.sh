#!/bin/bash
#SBATCH --partition=medium
#SBATCH --job-name=dyn_neuromod_grid
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --output=outputs/dyn_neuromod_5ht2a_%A_%a.out
#SBATCH --error=outputs/dyn_neuromod__5ht2a_%A_%a.err
#SBATCH --chdir=/network/iss/home/ivan.mindlin/
# Load Singularity module
module load singularity

# Define paths
SINGULARITY_IMAGE=/network/iss/home/ivan.mindlin/ubuntu_focal_conda.sif

# Run Python script inside Singularity container
singularity exec $SINGULARITY_IMAGE bash -c 'source /opt/anaconda/3/2023.07-2/base/etc/profile.d/conda.sh && conda activate fic && cd dyn_fic_dmf/python && python3 neuromod_script_dynamic5ht2a.py ' > ./dyn_neuromod_grid.out 