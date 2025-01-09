import sys
import os
import argparse
import math
import time
from scipy.signal import butter, lfilter
import fastdyn_fic_dmf as dmf
import numpy as np
from scipy.io import loadmat
from helper_functions import filter_bold
from multiprocessing import Pool
from scipy.stats import gamma  # Import gamma for entropy calculation


def calculate_gamma_entropy(node_index, rates):
    print(f"Processing node {node_index}")
    # Fit a gamma distribution to the data
    alpha, loc, beta = gamma.fit(rates)
    # Calculate the entropy of the gamma distribution
    entropy_value = gamma.entropy(a=alpha, loc=loc, scale=beta)
    return node_index, alpha, loc, beta, entropy_value

def grid_step(args):
    """
    Processes a single SEED and returns the entropy results.
    """
    SEED_tuple, params, nb_steps, burnout, overlap, isubfcd, a, b = args
    idx_SEED, SEED = SEED_tuple
    params['seed'] = SEED
    entropy_per_region = {}
    # Statistical FC
    params['G'] = loadmat('../matlab/Results/stat_fc/results_awake_stat_fc.mat')['minEstimatedG_Awake']
    params['alpha'] = loadmat('../matlab/Results/stat_fc/results_awake_stat_fc.mat')['minEstimatedY_Awake']
    params['J'] = params['alpha'] * params['G'] * params['C'].sum(axis=0).squeeze() + 1
    params["with_plasticity"] = False
    params["with_decay"] = False
    rates, _, _, _ = dmf.run(params, nb_steps)
    
    entropy_stat = []
    for node in range(params['N']):
        node_rates = rates[:, node]
        _, _, _, _, entropy = calculate_gamma_entropy(node, node_rates.reshape(rates.shape[0], -1))
        entropy_stat.append(entropy)
    stat_entropy = np.array(entropy_stat)
    # Dynamic FC
    params['G'] = loadmat('../matlab/Results/dyn_fc/results_awake_dyn_fc.mat')['minEstimatedG_Awake']
    params['lrj'] = loadmat('../matlab/Results/dyn_fc/results_awake_dyn_fc.mat')['minEstimatedY_Awake']
    DECAY = np.exp(a + np.log(params['lrj']) * b)
    params['taoj'] = DECAY
    params['alpha'] = 0.75
    params["with_plasticity"] = True
    params["with_decay"] = True
    params['J'] = params['alpha'] * params['G'] * params['C'].sum(axis=0).squeeze() + 1
    rates, _, _, _ = dmf.run(params, nb_steps)

    entropy_dyn = []
    for node in range(params['N']):
        node_rates = rates[:, node]
        _, _, _, _, entropy = calculate_gamma_entropy(node, node_rates.reshape(rates.shape[0], -1))
        entropy_dyn.append(entropy)
    dyn_entropy = np.array(entropy_dyn)

    return idx_SEED,stat_entropy,dyn_entropy
    

def main():
    
    # Prepare parameters and data
    C = loadmat('./data/DTI_fiber_consensus_HCP.mat')['connectivity'][:200, :200]
    C = 0.2 * C / np.max(C)
    params = dmf.default_params(C=C)

    isubfcd = np.triu_indices(C.shape[1], 1)
    params['N'] = C.shape[0]

    params["return_rate"] = True
    params["return_bold"] = True
    params["return_fic"] = True

    burnout = 7
    params["flp"] = 0.01
    params["fhp"] = 0.1
    params["wsize"] = 30
    overlap = 29
    params['TR'] = 2

    
    params['dtt'] = 0.001  # Assuming 'dtt' is defined; adjust as needed
    nb_steps = 50000

    fit_res = np.load("./data/fit_res_3-44.npy", allow_pickle=True)
    b = fit_res[0]  # First element is the slope
    a = fit_res[1]

    SEED_range = list(range(1, 101))  # SEEDs from 1 to 100

    NUM_CORES = 24  # Number of cores per node

    OBJ_RATE = 3.44
    params['obj_rate'] = OBJ_RATE

    task_args = [((idx_SEED, SEED), params.copy(), nb_steps, burnout, overlap, isubfcd, a, b)
                for idx_SEED, SEED in enumerate(SEED_range)]

    # Define the folder to save partial results
    results_folder = "./Results/EntropySimulation"
    os.makedirs(results_folder, exist_ok=True)

    # Process assigned SEEDs using multiprocessing Pool
    with Pool(processes=NUM_CORES) as pool:
        results = pool.map(grid_step, task_args)

    # Save partial results outside the pool
    file = os.path.join(results_folder, f"entropies.npy")
    np.save(file, results)
    

if __name__ == "__main__":
    main()
