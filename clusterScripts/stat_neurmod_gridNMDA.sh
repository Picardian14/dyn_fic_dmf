#!/bin/bash
#SBATCH --job-name=stat_neuromod_gridNMDA
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --output=outputs/stat_neuromodNMDA_%A_%a.out
#SBATCH --error=outputs/stat_neuromodNMDA_%A_%a.err
#SBATCH --chdir=/network/iss/home/ivan.mindlin/
# Load Singularity module
module load singularity

# Define paths
SINGULARITY_IMAGE=/network/iss/home/ivan.mindlin/ubuntu_focal_conda.sif

# Run Python script inside Singularity container
singularity exec $SINGULARITY_IMAGE bash -c 'source /opt/anaconda/3/2023.07-2/base/etc/profile.d/conda.sh && conda activate fic && cd dyn_fic_dmf/python && python3 neuromod_script_staticNMDA.py ' > ./stat_neuromod_gridNMDA.out 