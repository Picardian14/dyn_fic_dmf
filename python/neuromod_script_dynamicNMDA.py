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


C = loadmat('./data/DTI_fiber_consensus_HCP.mat')['connectivity'][:200, :200]
C = 0.2*C/np.max(C)
params = dmf.default_params(C=C)
RECEPTORS = np.load("./data/Schaeffer200-Tian/NMDA_ge179_hc29_galovic_schaeffer200.npy")[:200]
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
burnout = 10
params["flp"] = 0.008
params["fhp"] = 0.09
params["wsize"] = 30
overlap = 29
#nb_steps = 460000
#T = (nb_steps/params["TR"])*params["dtt"]
T = 250
params['TR'] = 0.72
nb_steps = int((T*params["TR"])/params["dtt"])
win_start = np.arange(0, T - burnout - params["wsize"], params["wsize"] - overlap)
nwins = len(win_start)
nints = len(isubfcd[0])
b_filter,a_filter = butter(2,np.array([0.01, 0.1])*2*params['TR'], btype='band')


# Load coefficients to estimte Decay with LR

fit_res = np.load("./data/fit_res_3-44.npy")
b = fit_res[0] # First element is the slope
a = fit_res[1]
triu_idx = np.triu_indices(C.shape[1],1)
params['N'] = C.shape[0]



#GAINE_range = np.arange(0,1,0.01)
GAINE_range = np.arange(0,1,0.01)
LR_range = np.logspace(0,2.46,10)
# Define the number of cores to use
NUM_CORES = 24

mean_fc_grid = np.zeros((len(LR_range),len(GAINE_range),params['N'],params['N']))
sim_fcds_grid = np.zeros((len(LR_range),len(GAINE_range),nwins-1,nwins-1))
mean_fr_grid = np.zeros((len(LR_range),len(GAINE_range), params['N']))
std_fr_grid = np.zeros((len(LR_range),len(GAINE_range), params['N']))
mean_fic_grid = np.zeros((len(LR_range),len(GAINE_range), params['N']))
std_fic_grid = np.zeros((len(LR_range),len(GAINE_range), params['N']))

G = 2.21
#LR = 44.3
def grid_step(args):
    LR_tuple,GAINE_tuple = args
    idx_GAINE,GAINE = GAINE_tuple[0],GAINE_tuple[1]    
    idx_LR,LR = LR_tuple[0],LR_tuple[1]    
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
    try:
        params['J'] = 0.75*params['G']*params['C'].sum(axis=0).squeeze() + 1    
        rates, _, bold, fic_t = dmf.run(params, nb_steps)     
        bold = bold[:, burnout:]
        # If a bold region has nans, replace with 0
        bold[np.isnan(bold)] = 0
        rates = rates[:,burnout*1000:]
        fic_t = fic_t[:,burnout*1000:]
        filt_bold = filter_bold(bold.T, params['flp'],params['fhp'], params['TR'])
        time_fc = compute_fcd(filt_bold, params["wsize"], overlap, isubfcd)
        # Replace 'compute_fcd' with the appropriate function or code that computes time_fc
        bold_fc = np.corrcoef(filt_bold.T)
        fcd = np.corrcoef(time_fc.T)    
        mean_firing_rates= np.mean(rates, axis=1)
        std_firing_rates= np.std(rates, axis=1)    
        mean_fic = np.mean(fic_t, axis=1)
        std_fic = np.std(fic_t, axis=1)
        return idx_LR,idx_GAINE, bold_fc, fcd,mean_firing_rates, std_firing_rates,mean_fic, std_fic
    except:
        print(f"Error with LR: {LR} and GAINE: {GAINE}")
        return None,None,None,None,None,None



from multiprocessing import Pool,Manager


# Define the number of cores to use

# Create a list of argument tuples for the nested loop function
args_list = [((idx_LR,LR),(idx_GAINE,GAINE))             
             for idx_GAINE,GAINE in enumerate(GAINE_range)
             for idx_LR, LR in enumerate(LR_range)]

manager = Manager()
results_list = manager.list()
# Create a pool of worker processes
with Pool(processes=NUM_CORES) as pool:
    # Map the nested loop function to the argument list across multiple processes
    results_list.extend(pool.map(grid_step, args_list))

#return idx_GAINE, mean_fc, sim_fcds,mean_firing_rates, std_firing_rates
for results in results_list:
    idx_LR = results[0]
    idx_GAINE = results[1]        
    mean_fc = results[2]
    sim_fcds = results[3] 
    mean_fr = results[4]  
    std_fr = results[5]
    mean_fic = results[6]
    std_fic = results[7]
    
    mean_fc_grid[idx_LR,idx_GAINE] = mean_fc
    sim_fcds_grid[idx_LR,idx_GAINE] = sim_fcds
    mean_fr_grid[idx_LR,idx_GAINE] = mean_fr
    std_fr_grid[idx_LR,idx_GAINE] = std_fr
    mean_fic_grid[idx_LR,idx_GAINE] = mean_fic
    std_fic_grid[idx_LR,idx_GAINE] = std_fic



import os

# Assuming these arrays are already populated with data

arrays_to_save = {
    'mean_fc_grid': mean_fc_grid,
    'sim_stds_grid': sim_fcds_grid,
    'mean_fr_grid': mean_fr_grid,
    'std_fr_grid': std_fr_grid,
    'mean_fic_grid': mean_fic_grid,
    'std_fic_grid': std_fic_grid
    
}


results_folder = "./Results/neuromod/dynamicNMDA"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)


# Save
for array_name, array_data in arrays_to_save.items():
    file_name = os.path.join(results_folder, f"{array_name}.npy")
    np.save(file_name, array_data)
