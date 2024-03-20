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
RECEPTORS = loadmat('data/SC_and_5ht2a_receptors.mat')['receptors']
RECEPTORS = RECEPTORS/max(RECEPTORS)-min(RECEPTORS)
RECEPTORS = RECEPTORS - max(RECEPTORS) + 1
params["receptors"] = RECEPTORS
# Main setup for this simulation
params["return_rate"] = True
params["return_bold"] = True
params["return_fic"] = True
# These are now default true
params["with_plasticity"] = True
params["with_decay"] = True

isubfcd = np.triu_indices(C.shape[1],1)
burnout = 7
params["flp"] = 0.01
params["fhp"] = 0.1
params["wsize"] = 30
overlap = 29
nb_steps = 100000
params['TR'] = 2.4
T = (nb_steps/params["TR"])*params["dtt"]
win_start = np.arange(0, T - burnout - params["wsize"], params["wsize"] - overlap)
nwins = len(win_start)
nints = len(isubfcd[0])
b_filter,a_filter = butter(2,np.array([0.01, 0.1])*2*params['TR'], btype='band')


# Load coefficients to estimte Decay with LR

coeffs = loadmat('./data/LinearFitCoefficients.mat')
a = coeffs['a'][0][0]
b = coeffs['b'][0][0]
triu_idx = np.triu_indices(C.shape[1],1)
params['N'] = C.shape[0]



GAINE_range = np.arange(0,1,0.01)
# Define the number of cores to use
NUM_CORES = 24

mean_fc_grid = np.zeros((len(GAINE_range),params['N'],params['N']))
sim_fcds_grid = np.zeros((len(GAINE_range),nwins-1,nwins-1))
mean_fr_grid = np.zeros((len(GAINE_range), params['N']))
std_fr_grid = np.zeros((len(GAINE_range), params['N']))


G = 2.21
LR = 44.3
def grid_step(args):
    GAINE_tuple = args
    idx_GAINE,GAINE = GAINE_tuple[0],GAINE_tuple[1]    
    params['G'] = G
    params['lrj'] = LR
    DECAY = np.exp(a+np.log(LR)*b)
    OBJ_RATE = 3.44    
    params['wgaine'] = GAINE
    params['wgaini'] = GAINE
    # Using heuristic linear rule 
    params['taoj'] = DECAY 
    params['obj_rate'] = OBJ_RATE
    #params['taoj'] = 210000
    params['J'] = 0.75*params['G']*params['C'].sum(axis=0).squeeze() + 1    
    rates, rates_inh, bold, fic_t = dmf.run(params, nb_steps)     
    bold = bold[:, burnout:]
    filt_bold = filter_bold(bold.T, params['flp'],params['fhp'], params['TR'])
    time_fc = compute_fcd(filt_bold, params["wsize"], overlap, isubfcd)
    # Replace 'compute_fcd' with the appropriate function or code that computes time_fc
    bold_fc = np.corrcoef(filt_bold.T)
    fcd = np.corrcoef(time_fc.T)    
    mean_firing_rates= np.mean(rates, axis=1)
    std_firing_rates= np.std(rates, axis=1)    

    return idx_GAINE, bold_fc, fcd,mean_firing_rates, std_firing_rates


from multiprocessing import Pool,Manager


# Define the number of cores to use

# Create a list of argument tuples for the nested loop function
args_list = [((idx_GAINE,GAINE))             
             for idx_GAINE,GAINE in enumerate(GAINE_range)]

manager = Manager()
results_list = manager.list()
# Create a pool of worker processes
with Pool(processes=NUM_CORES) as pool:
    # Map the nested loop function to the argument list across multiple processes
    results_list.extend(pool.map(grid_step, args_list))

#return idx_GAINE, mean_fc, sim_fcds,mean_firing_rates, std_firing_rates
for results in results_list:
    idx_GAINE = results[0]        
    mean_fc = results[1]
    sim_fcds = results[2] 
    mean_fr = results[3]  
    std_fr = results[4]
    
    mean_fc_grid[idx_GAINE] = mean_fc
    sim_fcds_grid[idx_GAINE] = sim_fcds
    mean_fr_grid[idx_GAINE] = mean_fr
    std_fr_grid[idx_GAINE] = std_fr


import os

# Assuming these arrays are already populated with data

arrays_to_save = {
    'mean_fc_grid': mean_fc_grid,
    'sim_stds_grid': sim_fcds_grid,
    'mean_fr_grid': mean_fr_grid,
    'std_fr_grid': std_fr_grid
    
}

results_folder = "./Results/neuromod/dynamic"

# Save
for array_name, array_data in arrays_to_save.items():
    file_name = os.path.join(results_folder, f"{array_name}.npy")
    np.save(file_name, array_data)
