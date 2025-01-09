#!/bin/bash
#SBATCH --partition=medium
#SBATCH --job-name=entropies
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output=outputs/entropies%A_%a.out
#SBATCH --error=outputs/entropies%A_%a.err
#SBATCH --chdir=/network/iss/home/ivan.mindlin/


# Load Singularity module
module load singularity

# Define paths
SINGULARITY_IMAGE=/network/iss/home/ivan.mindlin/ubuntu_focal_conda.sif

# Run Python script inside Singularity container with array job indexing
singularity exec $SINGULARITY_IMAGE bash -c 'source /opt/anaconda/3/2023.07-2/base/etc/profile.d/conda.sh && conda activate fic && cd dyn_fic_dmf/python && python3 entropy_simulations.py' > ./entropies.out