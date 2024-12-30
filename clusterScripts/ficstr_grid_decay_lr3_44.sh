#!/bin/bash
#SBATCH --job-name=ficstr_grid
#SBATCH --cpus-per-task=24
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=outputs/ficstr_%A_%a.out
#SBATCH --error=outputs/ficstr%A_%a.err
#SBATCH --chdir=/network/iss/home/ivan.mindlin/
#SBATCH --array=0-7
# Load Singularity module
module load singularity

# Define paths
SINGULARITY_IMAGE=/network/iss/home/ivan.mindlin/ubuntu_focal_conda.sif

# Run Python script inside Singularity container
singularity exec $SINGULARITY_IMAGE bash -c 'source /opt/anaconda/3/2023.07-2/base/etc/profile.d/conda.sh && conda activate fic && cd dyn_fic_dmf/python && python3 FicvsStr_3_44_Grid.py --task_idx ${SLURM_ARRAY_TASK_ID}' > ./ficvsstr_simulation_${SLURM_ARRAY_TASK_ID}.out
