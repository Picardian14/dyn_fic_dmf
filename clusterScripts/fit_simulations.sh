#!/bin/bash
#SBATCH --job-name=fit_simulation
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=outputs/fit_simulation%A_%a.out
#SBATCH --error=outputs/fit_simulation%A_%a.err

# Load Singularity module
module load singularity

# Define paths
SINGULARITY_IMAGE=/network/lustre/iss02/home/ivan.mindlin/ubuntu_focal_conda.sif

# Run Python script inside Singularity container
singularity exec $SINGULARITY_IMAGE bash -c 'source /opt/anaconda/3/2023.07-2/base/etc/profile.d/conda.sh && conda activate fic && cd dyn_fic_dmf/python && python3 fcd_simulations.py ' > ./fit_simulation.out 
