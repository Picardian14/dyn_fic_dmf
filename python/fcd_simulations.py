from scipy.signal import butter, lfilter
import fastdyn_fic_dmf as dmf
import numpy as np
import matplotlib.pyplot as plt
# Fetch default parameters
import tracemalloc
from scipy.io import loadmat
from scipy.stats import zscore, pearsonr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import os
import math
from helper_functions import filter_bold
#from mne.time_frequency import psd_array_multitaper


def compute_fcd(data, wsize, overlap, isubdiag):
    T, N = data.shape
    win_start = np.arange(0, T - params["wsize"] - 1, params["wsize"] - overlap)
    nwins = len(win_start)
    fcd = np.zeros((len(isubdiag[0]), nwins))
    for i in range(nwins):
        tmp = data[win_start[i]:win_start[i] + params["wsize"] + 1, :]
        cormat = np.corrcoef(tmp.T)
        fcd[:, i] = cormat[isubdiag[0],isubdiag[1]]
    return fcd


C = loadmat('data/SC_and_5ht2a_receptors.mat')['sc90']
C = 0.2*C/np.max(C)
params = dmf.default_params(C=C)

triu_idx = np.triu_indices(C.shape[1],1)
params['N'] = C.shape[0]
isubfcd = np.triu_indices(C.shape[1],1)


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
nb_steps = int(T*params['TR']/params['dtt'])
win_start = np.arange(0, T - burnout - params["wsize"], params["wsize"] - overlap)
nwins = len(win_start)
nints = len(isubfcd[0])

fit_res = np.load("./data/fit_res_3-44.npy")
b = fit_res[0] # First element is the slope
a = fit_res[1]

SEED_range = list(range(1,101))

mean_fc_grid = np.zeros((2,len(SEED_range),params['N'],params['N']))
sim_fcds_grid = np.zeros((2,len(SEED_range),nwins-1,nwins-1))
bold_grid = np.zeros((2,len(SEED_range), T-burnout, params['N']))

NUM_CORES=16

OBJ_RATE = 3.44    
# Using heuristic linear rule 
params['obj_rate'] = OBJ_RATE


def grid_step(args):
    SEED_tuple = args
    idx_SEED,SEED = SEED_tuple[0],SEED_tuple[1]    
    params['seed'] = SEED
    print(f"Doing Seed {SEED}")
    params['G'] = 1.74
    params['alpha'] = 0.67
    params['J'] = params['alpha']*params['G']*params['C'].sum(axis=0).squeeze() + 1    
    params["with_plasticity"] = False
    params["with_decay"] = False    
    rates, rates_inh, bold, fic_t = dmf.run(params, nb_steps)         
    bold = bold[:, burnout:]
    filt_bold = filter_bold(bold.T, params['flp'],params['fhp'], params['TR'])
    stat_bold = filt_bold
    time_fc = compute_fcd(filt_bold, params["wsize"], overlap, isubfcd)
    # Replace 'compute_fcd' with the appropriate function or code that computes time_fc
    stat_bold_fc = np.corrcoef(filt_bold.T)
    stat_fcd = np.corrcoef(time_fc.T)    

    params['G'] = 2.84
    params['lrj'] = 1.32
    DECAY = np.exp(a+np.log(params['lrj'])*b)
    params['taoj'] = DECAY 
    params['alpha'] = 0.75
    params["with_plasticity"] = True
    params["with_decay"] = True    
    params['J'] = params['alpha']*params['G']*params['C'].sum(axis=0).squeeze() + 1    
    rates, rates_inh, bold, fic_t = dmf.run(params, nb_steps)     
    bold = bold[:, burnout:]
    filt_bold = filter_bold(bold.T, params['flp'],params['fhp'], params['TR'])
    dyn_bold = filt_bold
    time_fc = compute_fcd(filt_bold, params["wsize"], overlap, isubfcd)
    # Replace 'compute_fcd' with the appropriate function or code that computes time_fc
    dyn_bold_fc = np.corrcoef(filt_bold.T)
    dyn_fcd = np.corrcoef(time_fc.T)    
 

    return idx_SEED, stat_bold_fc, stat_fcd,dyn_bold_fc, dyn_fcd,stat_bold, dyn_bold


from multiprocessing import Pool,Manager


# Define the number of cores to use

# Create a list of argument tuples for the nested loop function
args_list = [((idx_SEED,SEED))             
             for idx_SEED,SEED in enumerate(SEED_range)]

manager = Manager()
results_list = manager.list()
# Create a pool of worker processes
with Pool(processes=NUM_CORES) as pool:
    # Map the nested loop function to the argument list across multiple processes
    results_list.extend(pool.map(grid_step, args_list))

#return idx_SEED, mean_fc, sim_fcds,mean_firing_rates, std_firing_rates
for results in results_list:
    idx_SEED = results[0]        
    stat_bold_fc = results[1]
    stat_fcd = results[2] 
    dyn_bold_fc = results[3]  
    dyn_fcd = results[4]
    stat_bold = results[5]
    dyn_bold = results[6]
    
    mean_fc_grid[0,idx_SEED] = stat_bold_fc
    sim_fcds_grid[0,idx_SEED] = stat_fcd
    mean_fc_grid[1,idx_SEED] = dyn_bold_fc
    sim_fcds_grid[1,idx_SEED] = dyn_fcd
    bold_grid[0,idx_SEED] = stat_bold
    bold_grid[1,idx_SEED] = dyn_bold
    


import os

# Assuming these arrays are already populated with data

arrays_to_save = {
    'fcs_grid': mean_fc_grid,
    'fcds_grid': sim_fcds_grid,
    'bold_grid': bold_grid,
}

results_folder = "./Results/FittedSimulations"

# Save
for array_name, array_data in arrays_to_save.items():
    file_name = os.path.join(results_folder, f"{array_name}.npy")
    np.save(file_name, array_data)
