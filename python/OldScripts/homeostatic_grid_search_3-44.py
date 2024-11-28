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

#from mne.time_frequency import psd_array_multitaper


def compute_fcd(data, wsize, overlap, isubdiag):
    T, N = data.shape
    win_start = np.arange(0, T - wsize - 1, wsize - overlap)
    nwins = len(win_start)
    fcd = np.zeros((len(isubdiag[0]), nwins))
    for i in range(nwins):
        tmp = data[win_start[i]:win_start[i] + wsize + 1, :]
        cormat = np.corrcoef(tmp.T)
        fcd[:, i] = cormat[isubdiag[0],isubdiag[1]]
    return fcd


data_struct = loadmat('./data/ts_coma24_AAL_symm_withSC.mat')
data = np.zeros((13,90,192))
for i in range(13):
    data[i,:,:] = data_struct['timeseries_CNT24_symm'][0][i][:,:192]

C = loadmat('../SC_and_5ht2a_receptors.mat')['sc90']
C = 0.2*C/np.max(C)
params = dmf.default_params(C=C)
isubfcd = np.triu_indices(C.shape[1],1)
burnout = 7
flp = 0.01
fhp = 0.1
wsize = 30
overlap = 29
T = 250
win_start = np.arange(0, T - wsize, wsize - overlap)
nwins = len(win_start)
nints = len(isubfcd[0])
b_filter,a_filter = butter(2,np.array([0.01, 0.1])*2*params['TR'], btype='band')
NSUB = 13

emp_fcds = np.zeros((NSUB,4005,4005))
emp_fcs = np.zeros((NSUB,90,90))
for i in range(NSUB):
    bold = data[i]
    bold[:, (np.ceil(burnout / params['TR'])).astype(int):]    
    filt_bold = lfilter(b_filter,a_filter,bold)
    time_fc = compute_fcd(filt_bold.T, wsize, overlap, isubfcd)    
    bold_fc = np.corrcoef(filt_bold)
    fcd = np.corrcoef(time_fc)       
    emp_fcds[i] = fcd
    emp_fcs[i] = bold_fc

# Load coefficients to estimte Decay with LR

fit_res = np.load("./data/fit_res_3-44.npy")
b = fit_res[0] # First element is the slope
a = fit_res[1]
triu_idx = np.triu_indices(C.shape[1],1)
params['N'] = C.shape[0]

G_range = [1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5,2.6,2.7]
LR_range = np.arange(1,51,2)


params['TR'] = 2


nb_steps = int((data.shape[-1]-2*burnout)*params['TR']/params['dtt'])
#nb_steps = 100000

#G_range = [1,2]
#LR_range = [10,200]
# Define the number of cores to use
NUM_CORES = 24

fit_fc_grid = np.zeros((len(G_range),len(LR_range)))
fit_fcd_grid = np.zeros((len(G_range),len(LR_range)))
fc_grid = np.zeros((len(G_range),len(LR_range), params['N'],params['N']))
fcd_grid = np.zeros((len(G_range),len(LR_range), len(isubfcd[0]),len(isubfcd[0])))

burnout = 7



def grid_step(args):
    sim_fcds = np.zeros((NSUB,4005,4005))
    sim_fcs = np.zeros((NSUB,90,90))
    DECAY_tuple, LR_tuple = args
    idx_LR,LR = LR_tuple[0],LR_tuple[1]
    idx_DECAY,DECAY = DECAY_tuple[0],DECAY_tuple[1]
    print(f"Doing {DECAY} {LR}")
    DECAY = np.exp(a+np.log(LR)*b)
    OBJ_RATE = 3.44    
    params['lrj'] = LR        
    params['taoj'] = DECAY 
    params['obj_rate'] = OBJ_RATE    
    for G in G_range:
        params['G'] = G
        params['J'] = 0.75*params['G']*params['C'].sum(axis=0).squeeze() + 1
        rates, rates_inh, bold, fic_t = dmf.run(params, nb_steps,
                                                return_rate=True, return_bold=True, return_fic=True, 
                                                with_decay=True, with_plasticity=True)     
        
    print("Finished reptitions")
    mean_fc = np.mean(sim_fcs,axis=0)
    corr_to_sc = pearsonr(mean_fc[triu_idx[0],triu_idx[1]], C[triu_idx[0],triu_idx[1]])[0]        
    print("Calcualted corr to FC")
    mean_fcds = np.mean(sim_fcds,axis=0)
    ks, p = ks_2samp(mean_fcds.flatten(),np.mean(emp_fcds,axis=0).flatten())
    print("Calcualted sim fcd")

    
    return idx_DECAY,idx_LR, corr_to_sc,ks, mean_fc, mean_fcds


from multiprocessing import Pool,Manager


# Define the number of cores to use

# Create a list of argument tuples for the nested loop function
args_list = [((idx_G,G), (idx_LR,LR))
             for idx_G,G in enumerate(G_range)             
             for idx_LR,LR in enumerate(LR_range)]

manager = Manager()
results_list = manager.list()
# Create a pool of worker processes
with Pool(processes=NUM_CORES) as pool:
    # Map the nested loop function to the argument list across multiple processes
    results_list.extend(pool.map(grid_step, args_list))


for results in results_list:
    idx_G = results[0]    
    idx_LR = results[1]
    fit_fc = results[2]
    fit_fcd = results[3] 
    fc = results[4]  
    fcd = results[5]
    
    fit_fc_grid[idx_G,idx_LR] = fit_fc
    fit_fcd_grid[idx_G,idx_LR] = fit_fcd
    fc_grid[idx_G,idx_LR] = fc
    fcd_grid[idx_G,idx_LR] = fcd


import os

# Assuming these arrays are already populated with data

arrays_to_save = {
    'fit_fc_grid': fit_fc_grid,
    'fit_fcd_grid': fit_fcd_grid,
    'fc_grid': fc_grid,
    'fcd_grid': fcd_grid
    
}

results_folder = "./Results/fit_coma_cnt"

# Save
for array_name, array_data in arrays_to_save.items():
    file_name = os.path.join(results_folder, f"{array_name}-dynamic.npy")
    np.save(file_name, array_data)
