#!/bin/bash
#SBATCH --job-name=my_job_name
#SBATCH --output=output_%j.txt
#SBATCH --array=1-10%1

# Load Python module
# Activate the virtual environment (if using one)
source activate cloned_bstates

# Set the working directory
cd BrainStates_Causes

# Run Python script with input arguments
python3 example.py $SLURM_ARRAY_TASK_ID