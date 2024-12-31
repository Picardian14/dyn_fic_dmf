#!/usr/bin/env python3
import sys
import os
import argparse
import math
import time
from scipy.signal import butter, lfilter
import fastdyn_fic_dmf as dmf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from helper_functions import filter_bold
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
    Processes a single SEED and returns the results.
    """
    SEED_tuple, params, nb_steps, burnout, overlap, isubfcd, a, b = args
    idx_SEED, SEED = SEED_tuple
    params['seed'] = SEED
    #print(f"Processing Seed {SEED} (Index {idx_SEED})")
    
    # Statistical FC
    params['G'] = loadmat('../matlab/Results/stat_fc/results_awake_stat_fc.mat')['minEstimatedG_Awake']
    params['alpha'] = loadmat('../matlab/Results/stat_fc/results_awake_stat_fc.mat')['minEstimatedY_Awake']
    params['J'] = params['alpha'] * params['G'] * params['C'].sum(axis=0).squeeze() + 1
    params["with_plasticity"] = False
    params["with_decay"] = False
    rates, rates_inh, bold, fic_t = dmf.run(params, nb_steps)
    bold = bold[:, burnout:]
    filt_bold = filter_bold(bold.T, params['flp'], params['fhp'], params['TR'])
    stat_bold_fc = np.corrcoef(filt_bold.T)

    # Statistical FCD
    params['G'] = loadmat('../matlab/Results/stat_fcd/results_awake_stat_fcd.mat')['minEstimatedG_Awake']
    params['alpha'] = loadmat('../matlab/Results/stat_fcd/results_awake_stat_fcd.mat')['minEstimatedY_Awake']
    params['J'] = params['alpha'] * params['G'] * params['C'].sum(axis=0).squeeze() + 1
    rates, rates_inh, bold, fic_t = dmf.run(params, nb_steps)
    bold = bold[:, burnout:]
    filt_bold = filter_bold(bold.T, params['flp'], params['fhp'], params['TR'])
    stat_fcd = compute_fcd(filt_bold, params["wsize"], overlap, isubfcd, params)
    stat_fcd = np.corrcoef(stat_fcd.T)

    # Dynamic FC
    params['G'] = loadmat('../matlab/Results/dyn_fc/results_awake_dyn_fc.mat')['minEstimatedG_Awake']
    params['lrj'] = loadmat('../matlab/Results/dyn_fc/results_awake_dyn_fc.mat')['minEstimatedY_Awake']
    DECAY = np.exp(a + np.log(params['lrj']) * b)
    params['taoj'] = DECAY
    params['alpha'] = 0.75
    params["with_plasticity"] = True
    params["with_decay"] = True
    params['J'] = params['alpha'] * params['G'] * params['C'].sum(axis=0).squeeze() + 1
    rates, rates_inh, bold, fic_t = dmf.run(params, nb_steps)
    bold = bold[:, burnout:]
    filt_bold = filter_bold(bold.T, params['flp'], params['fhp'], params['TR'])
    dyn_bold_fc = np.corrcoef(filt_bold.T)

    # Dynamic FCD
    params['G'] = loadmat('../matlab/Results/dyn_fcd/results_awake_dyn_fcd.mat')['minEstimatedG_Awake']
    params['lrj'] = loadmat('../matlab/Results/dyn_fcd/results_awake_dyn_fcd.mat')['minEstimatedY_Awake']
    DECAY = np.exp(a + np.log(params['lrj']) * b)
    params['taoj'] = DECAY
    params['alpha'] = 0.75
    params["with_plasticity"] = True
    params["with_decay"] = True
    params['J'] = params['alpha'] * params['G'] * params['C'].sum(axis=0).squeeze() + 1
    rates, rates_inh, bold, fic_t = dmf.run(params, nb_steps)
    bold = bold[:, burnout:]
    filt_bold = filter_bold(bold.T, params['flp'], params['fhp'], params['TR'])
    dyn_fcd = compute_fcd(filt_bold, params["wsize"], overlap, isubfcd, params)
    dyn_fcd = np.corrcoef(dyn_fcd.T)

    return {
        'idx_SEED': idx_SEED,
        'stat_bold_fc': stat_bold_fc,
        'stat_fcd': stat_fcd,
        'dyn_bold_fc': dyn_bold_fc,
        'dyn_fcd': dyn_fcd,
        'stat_bold': filt_bold,
        'dyn_bold': filt_bold
    }

def integrate_results(total_tasks, results_folder, N, wsize, bold_length):
    """
    Integrates partial results from all tasks and saves the aggregated results.
    """
    print("Integrating partial results...")
    
    # Initialize arrays based on expected dimensions
    # Adjust the second dimension (SEEDs) based on your SEED_range
    SEED_count = 100  # Assuming SEED_range=1-100
    mean_fc_grid = np.zeros((2, SEED_count, N, N))
    sim_fcds_grid = np.zeros((2, SEED_count, wsize, wsize))  # Adjust dimensions as needed
    bold_grid = {
        'stat_bold': np.zeros((SEED_count, bold_length,N)),
        'dyn_bold': np.zeros((SEED_count, bold_length,N))
    }
    
    for task_idx in range(total_tasks):
        partial_file = os.path.join(results_folder, f"partial_result_{task_idx}.npy")
        if not os.path.exists(partial_file):
            print(f"Partial result file {partial_file} not found. Skipping.")
            continue
        partial_results = np.load(partial_file, allow_pickle=True)
        for partial in partial_results:
            idx_SEED = partial['idx_SEED']
            stat_bold_fc = partial['stat_bold_fc']
            stat_fcd = partial['stat_fcd']
            dyn_bold_fc = partial['dyn_bold_fc']
            dyn_fcd = partial['dyn_fcd']
            stat_bold = partial['stat_bold']
            dyn_bold = partial['dyn_bold']

            mean_fc_grid[0, idx_SEED] = stat_bold_fc
            sim_fcds_grid[0, idx_SEED] = stat_fcd
            mean_fc_grid[1, idx_SEED] = dyn_bold_fc
            sim_fcds_grid[1, idx_SEED] = dyn_fcd
            bold_grid['stat_bold'][idx_SEED] = stat_bold
            bold_grid['dyn_bold'][idx_SEED] = dyn_bold

    # Save integrated results
    arrays_to_save = {
        'fcs_grid': mean_fc_grid,
        'fcds_grid': sim_fcds_grid,
        'bold_grid_stat': bold_grid['stat_bold'],
        'bold_grid_dyn': bold_grid['dyn_bold'],
    }

    final_results_folder = "./Results/FittedSimulations"
    os.makedirs(final_results_folder, exist_ok=True)

    for array_name, array_data in arrays_to_save.items():
        file_name = os.path.join(final_results_folder, f"{array_name}.npy")
        np.save(file_name, array_data)
        print(f"Saved integrated {array_name} to {file_name}")

def main():
    parser = argparse.ArgumentParser(description="Distributed Simulation Script")
    parser.add_argument('--task_idx', type=int, required=True, help='Task index (0 to total_tasks-1)')
    args = parser.parse_args()

    task_idx = args.task_idx
    total_tasks = 8  # Fixed based on SLURM array=0-7

    # Prepare parameters and data
    C = loadmat('./data/DTI_fiber_consensus_HCP.mat')['connectivity'][:200, :200]
    C = 0.2 * C / np.max(C)
    params = dmf.default_params(C=C)

    triu_idx = np.triu_indices(C.shape[1], 1)
    params['N'] = C.shape[0]
    isubfcd = np.triu_indices(C.shape[1], 1)

    # Main setup for this simulation
    params["return_rate"] = True
    params["return_bold"] = True
    params["return_fic"] = True

    burnout = 7
    params["flp"] = 0.01
    params["fhp"] = 0.1
    params["wsize"] = 30
    overlap = 29
    params['TR'] = 2

    T = 250
    params['dtt'] = 0.001  # Assuming 'dtt' is defined; adjust as needed
    nb_steps = int(T * params['TR'] / params['dtt'])
    win_start = np.arange(0, T - burnout - params["wsize"], params["wsize"] - overlap)
    nwins = len(win_start)
    nints = len(isubfcd[0])

    fit_res = np.load("./data/fit_res_3-44.npy", allow_pickle=True)
    b = fit_res[0]  # First element is the slope
    a = fit_res[1]

    SEED_range = list(range(1, 101))  # SEEDs from 1 to 100

    NUM_CORES = 24  # Number of cores per node

    OBJ_RATE = 3.44
    params['obj_rate'] = OBJ_RATE

    # Determine the subset of SEEDs for this task based on task_idx
    chunk_size = math.ceil(len(SEED_range) / total_tasks)
    start_idx = task_idx * chunk_size
    end_idx = min(start_idx + chunk_size, len(SEED_range))
    task_seeds = SEED_range[start_idx:end_idx]
    task_indices = list(range(start_idx, end_idx))
    task_args = [((idx_SEED, SEED), params.copy(), nb_steps, burnout, overlap, isubfcd, a, b)
                for idx_SEED, SEED in zip(task_indices, task_seeds)]

    # Define the folder to save partial results
    partial_results_folder = "./Results/PartialResults"
    os.makedirs(partial_results_folder, exist_ok=True)

    # Process assigned SEEDs using multiprocessing Pool
    with Pool(processes=NUM_CORES) as pool:
        results = pool.map(grid_step, task_args)

    # Save partial results outside the pool
    partial_file = os.path.join(partial_results_folder, f"partial_result_{task_idx}.npy")
    np.save(partial_file, results)
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

        # Integrate results
        integrate_results(total_tasks, partial_results_folder, params['N'], nwins-1, T-burnout)
        print("Integration completed.")

if __name__ == "__main__":
    main()
