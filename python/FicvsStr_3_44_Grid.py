#!/usr/bin/env python3
import sys
import os
import argparse
import math
import time
from scipy.io import loadmat
import fastdyn_fic_dmf as dmf
import numpy as np
from multiprocessing import Pool

def compute_fcd(data, wsize, overlap, isubdiag, params):
    T, N = data.shape
    win_start = np.arange(0, T - params["wsize"] - 1, params["wsize"] - overlap)
    nwins = len(win_start)
    fcd = np.zeros((len(isubdiag[0]), nwins))
    for i in range(nwins):
        tmp = data[win_start[i]:win_start[i] + params["wsize"] + 1, :]
        cormat = np.corrcoef(tmp.T)
        fcd[:, i] = cormat[isubdiag[0], isubdiag[1]]
    return fcd

def grid_step(args):
    """
    Processes a single (DECAY, LR) pair and returns the results.
    """
    ((idx_DECAY, DECAY), (idx_LR, LR), params, nb_steps, burnout, overlap, isubfcd, G_range, N_RANDOMIZATIONS, node_strength, post_burnout_steps) = args
    print(f"Starting grid step: DECAY={DECAY}, LR={LR}")
    OBJ_RATE = 3.44
    params['lrj'] = LR
    params['taoj'] = DECAY
    params['obj_rate'] = OBJ_RATE

    # Prepare arrays for storing results of this DECAY/LR pair
    # Shape: (N_RANDOMIZATIONS, len(G_range), post_burnout_steps)
    all_random_fic_cor = np.zeros((N_RANDOMIZATIONS, len(G_range), post_burnout_steps), dtype=np.float32)

    for r in range(N_RANDOMIZATIONS):
        for idx_G, G_val in enumerate(G_range):
            params['G'] = G_val
            params['seed'] = idx_G + 1000 * r
            reference_fic = 0.75 * params['G'] * params['C'].sum(axis=0).squeeze() + 1
            # Create random vector that is in the same range as the reference fic
            params['J'] = np.random.rand(params['C'].shape[0]) * reference_fic.max()

            rates, _, _, fic_t = dmf.run(params, nb_steps)

            # Extract the post-burnout part of the FIC timeseries
            # fic_t shape: (nodes, time)
            burnout_steps = int(math.ceil(burnout * 1000))
            fic_t_post_burnout = fic_t[:, burnout_steps:]

            # Compute correlation at each time step
            # For each time t, we correlate fic_t_post_burnout[:, t] with node_strength
            for t in range(post_burnout_steps):
                # Compute correlation with node_strength
                if np.std(fic_t_post_burnout[:, t]) == 0 or np.std(node_strength) == 0:
                    cor = 0  # Handle zero variance
                else:
                    cor = np.corrcoef(fic_t_post_burnout[:, t], node_strength)[0, 1]
                all_random_fic_cor[r, idx_G, t] = cor

    return idx_DECAY, idx_LR, all_random_fic_cor

def integrate_results(total_tasks, results_folder, fic_cor_timeseries_grid_shape, output_folder):
    """
    Integrates partial results from all tasks and saves the aggregated results.
    """
    print("Integrating partial results...")
    
    fic_cor_timeseries_grid = np.zeros(fic_cor_timeseries_grid_shape, dtype=np.float32)
    
    for task_idx in range(total_tasks):
        partial_file = os.path.join(results_folder, f"partial_result_{task_idx}.npy")
        if not os.path.exists(partial_file):
            print(f"Partial result file {partial_file} not found. Skipping.")
            continue
        partial_results = np.load(partial_file, allow_pickle=True)
        for partial in partial_results:
            idx_DECAY = partial['idx_DECAY']
            idx_LR = partial['idx_LR']
            all_random_fic_cor = partial['all_random_fic_cor']  # shape (N_RANDOMIZATIONS, len(G_range), post_burnout_steps)

            fic_cor_timeseries_grid[idx_DECAY, idx_LR] = all_random_fic_cor

    # Save integrated results
    arrays_to_save = {
        'fic_cor_timeseries_grid': fic_cor_timeseries_grid
    }

    os.makedirs(output_folder, exist_ok=True)

    for array_name, array_data in arrays_to_save.items():
        file_name = os.path.join(output_folder, f"{array_name}.npy")
        np.save(file_name, array_data)
        print(f"Saved integrated {array_name} to {file_name}")

def main():
    parser = argparse.ArgumentParser(description="Distributed FIC Correlation Simulation Script")
    parser.add_argument('--task_idx', type=int, required=True, help='Task index (0 to total_tasks-1)')
    args = parser.parse_args()

    task_idx = args.task_idx
    total_tasks = 8  # Based on SLURM array=0-7

    # Prepare parameters and data
    C = loadmat('./data/DTI_fiber_consensus_HCP.mat')['connectivity'][:200, :200]
    C = 0.2 * C / np.max(C)
    params = dmf.default_params(C=C)

    triu_idx = np.triu_indices(C.shape[1], 1)
    params['N'] = C.shape[0]
    isubfcd = np.triu_indices(C.shape[1], 1)

    # Main simulation setup
    params["return_rate"] = True
    params["return_bold"] = False
    params["return_fic"] = True  # Ensure fic_t is returned
    params["with_plasticity"] = True
    params["with_decay"] = True

    G_range = np.arange(0, 8.5, 0.5)
    LR_range = np.logspace(0, 3, 100)
    DECAY_range = np.logspace(2, 6, 110)

    burnout = 7
    nb_steps = 50000
    NUM_CORES = 24
    N_RANDOMIZATIONS = 1

    node_strength = C.sum(axis=0)
    # Number of time steps after the burnout period
    post_burnout_steps = nb_steps - burnout * 1000

    # Create a list of argument tuples for the nested loop function
    args_list = [((idx_DECAY, DECAY), (idx_LR, LR), params.copy(), nb_steps, burnout, 29, isubfcd, G_range, N_RANDOMIZATIONS, node_strength, post_burnout_steps)
                 for idx_DECAY, DECAY in enumerate(DECAY_range)
                 for idx_LR, LR in enumerate(LR_range)]

    # Determine the subset of args_list for this task based on task_idx
    total_args = len(args_list)
    chunk_size = math.ceil(total_args / total_tasks)
    start_idx = task_idx * chunk_size
    end_idx = min(start_idx + chunk_size, total_args)
    task_args = args_list[start_idx:end_idx]

    # Define the folder to save partial results
    partial_results_folder = "./Results/Figure1/FicvsStr3-44-Grid/PartialResults"
    os.makedirs(partial_results_folder, exist_ok=True)

    # Process assigned (DECAY, LR) pairs using multiprocessing Pool
    with Pool(processes=NUM_CORES) as pool:
        results = pool.map(grid_step, task_args)

    # Prepare data to save
    # Each result is a tuple: (idx_DECAY, idx_LR, all_random_fic_cor)
    partial_results = []
    for res in results:
        idx_DECAY, idx_LR, all_random_fic_cor = res
        partial_results.append({
            'idx_DECAY': idx_DECAY,
            'idx_LR': idx_LR,
            'all_random_fic_cor': all_random_fic_cor
        })

    # Save partial results outside the pool
    partial_file = os.path.join(partial_results_folder, f"partial_result_{task_idx}.npy")
    np.save(partial_file, partial_results)
    print(f"Task {task_idx}: Saved partial results to {partial_file}")

    # If this is the designated integrator task, perform integration
    # For example, task_idx=0 acts as the integrator
    if task_idx == 0:
        print("Integrator task started. Waiting for all partial results...")
        expected_files = [os.path.join(partial_results_folder, f"partial_result_{i}.npy") for i in range(total_tasks)]
        
        while True:
            existing_files = [f for f in expected_files if os.path.exists(f)]
            if len(existing_files) >= total_tasks:
                print("All partial results found. Proceeding to integrate.")
                break
            else:
                print(f"Waiting for partial results... ({len(existing_files)}/{total_tasks} files found)")
                time.sleep(60)  # Wait for 60 seconds before checking again

        # Define the shape for fic_cor_timeseries_grid
        fic_cor_timeseries_grid_shape = (
            len(DECAY_range),
            len(LR_range),
            N_RANDOMIZATIONS,
            len(G_range),
            post_burnout_steps
        )

        # Define the output folder
        output_folder = "./Results/Figure1/FicvsStr3-44-Grid"

        # Integrate results
        integrate_results(total_tasks, partial_results_folder, fic_cor_timeseries_grid_shape, output_folder)
        print("Integration completed.")

if __name__ == "__main__":
    main()
