#!/bin/bash
#SBATCH --job-name=IntegrateBayesianRes
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --chdir=/network/iss/home/ivan.mindlin/dyn_fic_dmf/matlab
ml matlab/R2022b
matlab -nodisplay -nosplash -nodesktop -r "run('ObjFunctionEstimatiom.m'); exit;"