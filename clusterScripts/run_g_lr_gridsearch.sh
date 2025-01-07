#!/bin/bash
#SBATCH --job-name=g_lr_dyn_grid
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --output=outputs/g_lr_dyn_%A_%a.out
#SBATCH --error=outputs/g_lr_dyn%A_%a.err
#SBATCH --chdir=/network/iss/home/ivan.mindlin/
#SBATCH --array=0-23
# Load Singularity module
module load singularity

# Define paths
SINGULARITY_IMAGE=/network/iss/home/ivan.mindlin/ubuntu_focal_conda.sif

# Run Python script inside Singularity container
#singularity exec $SINGULARITY_IMAGE bash -c 'source /opt/anaconda/3/2023.07-2/base/etc/profile.d/conda.sh && conda activate fic && cd dyn_fic_dmf/python && python3 G_LR_gridsearch_script_dynamic.py --task_idx ${SLURM_ARRAY_TASK_ID}' > ./g_lr.out 
singularity exec --bind /network/iss/cohen/data/Ivan:/network/iss/cohen/data/Ivan:rw $SINGULARITY_IMAGE bash -c 'source /opt/anaconda/3/2023.07-2/base/etc/profile.d/conda.sh && conda activate fic && cd dyn_fic_dmf/python && python3 G_LR_gridsearch_script_dynamic.py --task_idx ${SLURM_ARRAY_TASK_ID}' > ./g_lr.out 