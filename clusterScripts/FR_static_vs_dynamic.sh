#!/bin/bash
#SBATCH --job-name=fr_statvsdyn
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=24
#SBATCH --time=72:00:00
#SBATCH --mem=90G
#SBATCH --output=outputs/fr_statvsdyn_%A_%a.out
#SBATCH --error=outputs/fr_statvsdyn_%A_%a.err
#SBATCH --chdir=/network/iss/home/ivan.mindlin/

# Load Singularity module
module load singularity

# Define paths
SINGULARITY_IMAGE=/network/iss/home/ivan.mindlin/ubuntu_focal_conda.sif

# Run Python script inside Singularity container
singularity exec $SINGULARITY_IMAGE bash -c 'source /opt/anaconda/3/2023.07-2/base/etc/profile.d/conda.sh && conda activate fic && cd dyn_fic_dmf/python && python3 FR_static_dynamic_compare.py ' > ./fr_stat-vs-dyn.out 

