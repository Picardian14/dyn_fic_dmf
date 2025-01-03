#!/usr/bin/env python3
import sys
import os
import argparse
import math
import time
import numpy as np
from scipy.io import loadmat
from multiprocessing import Pool

# Custom modules (as in your snippet)
import fastdyn_fic_dmf as dmf
from helper_functions import filter_bold

################################################################################
# Helper functions
################################################################################

def compute_fc(data):
    """
    Compute FC by correlating BOLD (or rate) time series.
    data shape: (time, nodes).
    Returns an (N x N) correlation matrix.
    """
    return np.corrcoef(data.T)  # shape [N, N]

def compute_fcd(data, wsize, overlap, isubdiag):
    """
    Compute FCD by sliding a window of length 'wsize' with 'overlap' into 'data'.
    
    data shape: (time, nodes).
    wsize, overlap: integer time points.
    isubdiag: np.triu_indices(N, 1) or similar.

    Returns a 2D array:
        shape [num_windows, number_of_corr_pairs]
    where number_of_corr_pairs = len(isubdiag[0]).
    """
    T, N = data.shape
    step = wsize - overlap
    win_starts = np.arange(0, T - wsize + 1, step)
    
    fcd_mat = []
    for start in win_starts:
        window_data = data[start:start + wsize, :]
        cormat = np.corrcoef(window_data.T)
        fcd_mat.append(cormat[isubdiag])
    fcd_mat = np.corrcoef(np.array(fcd_mat))  # shape [num_windows, n_subdiag]
    return fcd_mat  # shape [num_windows, n_subdiag]



def simulate_one_seed(args):
    """
    Run the DMF model for a single seed, given alpha and G (plus a,b for DECAY),
    and return (FC, FCD).
    """
    (params_base, nb_steps, burnout, wsize, overlap, seed_id, a, b, isubdiag) = args

    # Copy base params so we don't mutate them
    params = params_base.copy()
    params['seed'] = seed_id
    
    # Because we want to see plastic changes
    params["with_plasticity"] = False
    params["with_decay"]      = False
    
    # (Re)compute 'J' after setting alpha, G, etc.
    # If alpha is always 0.75 or you prefer a different logic, do it here:
    
    params['J'] = params['alpha'] * params['G'] * params['C'].sum(axis=0).squeeze() + 1

    # Run DMF
    # Returns: (rates, bold, fic, etc.) -- adjust if your function differs
    _, _, bold, _ = dmf.run(params, nb_steps)
    
    # Burn out the initial transients
    bold_post = bold[:, burnout:]  # shape [N, T-burnout]
    
    # If we need to filter the BOLD, do it here
    # Convert to shape [time, nodes] for correlation
    bold_post = bold_post.T  # shape [T-burnout, N]
    bold_filt = filter_bold(bold_post, params['flp'], params['fhp'], params['TR'])

    # Compute FC
    fc_seed = compute_fc(bold_filt)  # shape [N, N]

    # Compute FCD
    fcd_seed = compute_fcd(bold_filt, wsize, overlap, isubdiag)  
    # shape [num_windows, n_subdiag]

    return fc_seed, fcd_seed

def grid_step(args):
    """
    Process a single (alpha, G) pair by running multiple seeds.
    Return:
      {
        'idx_alpha': idx_alpha,
        'idx_g':  idx_g,
        'FC_avg':  (N x N),
        'FCD_stacked': (n_seeds x num_windows x n_subdiag)
      }
    """
    (idx_alpha, alpha_val, idx_g, G_val,
     params, nb_steps, burnout, wsize, overlap, n_seeds,
     a, b, isubdiag) = args

    # Set the alpha & G in the param dictionary
    params['alpha'] = alpha_val
    params['G']   = G_val
    
    N = params['N']

    # We'll sum FCs to get an average later
    fc_sum = np.zeros((N, N), dtype=np.float32)
    fcd_list = []

    # Prepare a local list of seeds
    # Example: we might do seeds = range(n_seeds), or something more elaborate
    # For demonstration, we just do seeds 0..(n_seeds-1)
    seed_list = range(n_seeds)
    
    # Build arguments for simulate_one_seed
    simulate_args = []
    for seed_id in seed_list:
        # Some unique seed scheme:
        # e.g. seed_in = seed_id + 1000 * idx_alpha + 10000 * idx_g
        seed_in = seed_id + idx_alpha + 2 * idx_g
        simulate_args.append((
            params,           # base param
            nb_steps,
            burnout,
            wsize,
            overlap,
            seed_in,
            a,
            b,
            isubdiag
        ))
    
    # Run seeds in parallel
    NWORKERS = 16
    fcs = []
    with Pool(NWORKERS) as local_pool:
        results = local_pool.map(simulate_one_seed, simulate_args)
    # Kill Pool
    local_pool.close()

    # Aggregate
    for fc_seed, fcd_seed in results:
        fc_sum += fc_seed
        fcd_list.append(fcd_seed)
        fcs.append(fc_seed)

    # Average FC
    
    fc_avg = fc_sum / n_seeds
    
    # Stack FCD => shape [n_seeds, num_windows, n_subdiag]
    fcd_stacked = np.stack(fcd_list, axis=0)

    return {
        'idx_alpha': idx_alpha,
        'idx_g':  idx_g,
        'FC_avg': fc_avg,
        'FCD_stacked': fcd_stacked,
        'fcs': fcs
    }

################################################################################
# Integration of partial results
################################################################################

def integrate_results(total_tasks, results_folder,
                      nalpha, nG, n_seeds, output_folder):
    """
    Loads partial results from partial_result_0..(total_tasks-1).npy
    and constructs final arrays:
      FC_grid:  (nalpha, nG, N, N)
      FCD_grid: (nalpha, nG, n_seeds, num_windows, n_subdiag)

    Then saves them to output_folder.
    """
    print("[integrate_results] Integrating partial results...")

    FC_grid = None
    FCD_grid = None
    loaded_something = False

    for task_idx in range(total_tasks):
        partial_file = os.path.join(results_folder, f"partial_result_{task_idx}.npy")
        if not os.path.exists(partial_file):
            print(f"  [Warning] partial file not found: {partial_file}")
            continue
        
        # Each partial is a list of dicts: { 'idx_alpha', 'idx_g', 'FC_avg', 'FCD_stacked' }
        partial_data = np.load(partial_file, allow_pickle=True)
        
        for item in partial_data:
            idx_alpha = item['idx_alpha']
            idx_g  = item['idx_g']
            fc_avg = item['FC_avg']       # shape [N, N]
            fcd_st = item['FCD_stacked']  # shape [n_seeds, num_windows, n_subdiag]

            if not loaded_something:
                N = fc_avg.shape[0]
                num_windows = fcd_st.shape[1]
                n_subdiag  = fcd_st.shape[2]

                FC_grid  = np.zeros((nalpha, nG, N, N), dtype=np.float32)
                FCD_grid = np.zeros((nalpha, nG, n_seeds, num_windows, num_windows), dtype=np.float32)
                loaded_something = True

            FC_grid[idx_alpha, idx_g]  = fc_avg
            FCD_grid[idx_alpha, idx_g] = fcd_st

    # Save final results
    os.makedirs(output_folder, exist_ok=True)
    if FC_grid is not None:
        np.save(os.path.join(output_folder, "FC_grid.npy"), FC_grid)
        print("[integrate_results] Saved FC_grid")
    if FCD_grid is not None:
        np.save(os.path.join(output_folder, "FCD_grid.npy"), FCD_grid)
        print("[integrate_results] Saved FCD_grid")

    print("[integrate_results] Done.")

################################################################################
# Main script
################################################################################

def main():
    parser = argparse.ArgumentParser(description="Grid search over alpha & G, computing FC & FCD.")
    parser.add_argument('--task_idx', type=int, required=True,
                        help='Task index (0..total_tasks-1 for SLURM array).')
    args = parser.parse_args()

    # Slurm-like config
    total_tasks = 8
    task_idx = args.task_idx

        # Load structural connectivity
    C = loadmat('./data/DTI_fiber_consensus_HCP.mat')['connectivity'][:200,:200]
    C = 0.2 * C / np.max(C)

    # Base DMF params
    params = dmf.default_params(C=C)
    params['N'] = C.shape[0]
    N = params['N']

    # Filtering params (adapt to your usage)
    params["flp"] = 0.01
    params["fhp"] = 0.1
    params["TR"]  = 2

    # For windowed FCD
    wsize   = 30
    overlap = 29

    # Burnout in time points (or directly # of samples)
    burnout = 7  # if your run uses 1 step = 1 ms, might be burnout * 1000
                    # but adapt to how your model is defined

    # Total simulation time in TRs:
    T = 250
    # We assume dtt from your snippet:
    params['dtt'] = 0.001
    nb_steps = int(T * params["TR"] / params["dtt"])

    # Indices for the upper triangular part (for FC or FCD)
    isubdiag = np.triu_indices(N, 1)

    # Load slope and intercept for DECAY
    fit_res  = np.load("./data/fit_res_3-44.npy", allow_pickle=True)
    b = fit_res[0]  # slope
    a = fit_res[1]  # intercept

    # Grid definitions
    nalpha = 110
    ALPHA_range = np.logspace(0, 3, nalpha)   # from 1 to 1000
    nG  = 100
    G_range = np.linspace(0.1, 16, nG)  # from 0.1 to 16

    # Seeds per (alpha, G)
    n_seeds = 16

    # Build the entire list of (alpha, G) pairs
    all_args = []
    for idx_alpha, alpha_val in enumerate(ALPHA_range):
        for idx_g, G_val in enumerate(G_range):
            # We pass everything needed, including a,b
            all_args.append((
                idx_alpha, alpha_val,
                idx_g,  G_val,
                params.copy(),
                nb_steps,
                burnout,
                wsize,
                overlap,
                n_seeds,
                a, b,
                isubdiag
            ))

    # Distribute (alpha, G) pairs among tasks
    total_pairs = len(all_args)
    chunk_size = math.ceil(total_pairs / total_tasks)
    start_idx = task_idx * chunk_size
    end_idx   = min(start_idx + chunk_size, total_pairs)
    task_args = all_args[start_idx:end_idx]

    # Folder for partial results
    partial_results_folder = "./Results/Partial_Grid_alpha_G"
    os.makedirs(partial_results_folder, exist_ok=True)

    # Each task processes its chunk of (alpha, G) pairs in SERIAL,
    # but seeds are parallelized in grid_step().
    results = []
    for arg_tuple in task_args:
        # grid_step() itself parallelizes across seeds
        result = grid_step(arg_tuple)
        results.append(result)

    # Save partial results
    partial_file = os.path.join(partial_results_folder, f"partial_result_{task_idx}.npy")
    np.save(partial_file, results)
    print(f"[main] Task {task_idx} saved partial results: {partial_file}")

    # Integrator on task_idx=0
    if task_idx == 0:
        print("[main] Integrator waiting for other tasks...")
        expected_files = [
            os.path.join(partial_results_folder, f"partial_result_{i}.npy")
            for i in range(total_tasks)
        ]
        while True:
            existing = [f for f in expected_files if os.path.exists(f)]
            if len(existing) == total_tasks:
                print("[main] All partial results found. Proceeding to integration.")
                break
            else:
                print(f"[main] {len(existing)}/{total_tasks} partial files found, waiting...")
                time.sleep(30)

        # Integrate them into final arrays
        output_folder = "./Results/FC_FCD_Grid"
        integrate_results(total_tasks,
                          partial_results_folder,
                          nalpha, nG, n_seeds,
                          output_folder)
        print("[main] Integration completed.")

if __name__ == "__main__":
    main()
