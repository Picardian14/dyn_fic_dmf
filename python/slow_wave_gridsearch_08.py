import fastdyn_fic_dmf as dmf
import numpy as np
import matplotlib.pyplot as plt
# Fetch default parameters
from scipy.io import loadmat
from scipy.stats import  pearsonr
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def get_max_freq_and_power(rates):
    freqs, psd = welch(rates, fs=1000,axis=1,nperseg=4*1000,noverlap=2*1000)
    max_freq_id = np.argmax(psd[:,:100],axis=1)
    max_freqs = freqs[max_freq_id]
    max_power = np.max(psd[:,:100],axis=1)
    return max_freqs, max_power,freqs,psd

nb_steps = 50000
C = loadmat('./data/DTI_fiber_consensus_HCP.mat')['connectivity'][:200, :200]
# Load coefficients to estimte Decay with LR
fit_res = np.load("./data/fit_res_3-44.npy")
b = fit_res[0] # First element is the slope
a = fit_res[1]
C = 0.2*C/np.max(C)
triu_idx = np.triu_indices(C.shape[1],1)
burnout = 7
params = dmf.default_params(C=C)
params['N'] = C.shape[0]



G_max = 8
G_step = 0.25
G_range = np.arange(1,G_max,G_step)
LR_range = np.logspace(0, 4,100)


# Define the number of cores to use
NUM_CORES = 32
NREP = 8
max_freq_grid = np.zeros((NREP,len(G_range),len(LR_range)))
max_power_grid = np.zeros((NREP,len(G_range),len(LR_range)))
std_slow_grid = np.zeros((NREP,len(G_range),len(LR_range)))
corr_to_sc_grid = np.zeros((NREP,len(G_range),len(LR_range)))
homeostatic_fittness_grid = np.zeros((NREP,len(G_range),len(LR_range),params['N']))
rates_grid = np.zeros((NREP,len(G_range),len(LR_range),params['N']))
fic_t_grid = np.zeros((NREP,len(G_range),len(LR_range),params['N']))

params['with_plasticity'] = True
params['with_decay'] = True
params['return_bold'] = False
params['return_rate'] = True
params['return_fic'] = True


def grid_step(args):
    all_max_freq = np.zeros((NREP))
    all_max_power = np.zeros((NREP))
    all_corr = np.zeros((NREP))
    all_homfit = np.zeros((NREP, params['N']))
    all_rates = np.zeros((NREP, params['N']))
    all_fic_t = np.zeros((NREP, params['N'] ))


    G_tuple, LR_tuple = args
    idx_LR,LR = LR_tuple[0],LR_tuple[1]
    idx_G,G = G_tuple[0],G_tuple[1]    
    DECAY = np.exp(a+np.log(LR)*b)
    OBJ_RATE = 3.44        
    params['lrj'] = LR
    params['G'] = G
    # Using heuristic linear rule 
    params['taoj'] = DECAY 
    params['obj_rate'] = OBJ_RATE
    #params['taoj'] = 210000
    params['J'] = 0.75*params['G']*params['C'].sum(axis=0).squeeze() + 1
    
    for idx in range(NREP):
        rates, _, _, fic_t = dmf.run(params, nb_steps)        
        rates = rates[:, np.ceil(burnout * 1000).astype(int):]    
        fic_t = fic_t[:, np.ceil(burnout * 1000).astype(int):]    
        rates_fc = np.corrcoef(rates)
        all_rates[idx,:] = np.mean(rates, axis=1)
        all_fic_t[idx,:] = np.mean(fic_t, axis=1)
        corr_to_sc = pearsonr(rates_fc[triu_idx[0],triu_idx[1]], C[triu_idx[0],triu_idx[1]])[0]
        all_corr[idx] = corr_to_sc
        max_freq,max_power,_,_ = get_max_freq_and_power(rates)        
        all_max_freq[idx] = np.mean(max_freq)
        all_max_power[idx] = np.mean(max_power)
        homeostatic_fittness =  OBJ_RATE - np.mean(rates,axis=1)  
        all_homfit[idx, :] = np.mean(homeostatic_fittness)
    max_freq = all_max_freq
    max_power = all_max_power
    corr_to_sc = all_corr
    homeostatic_fittness = all_homfit
    rates = all_rates
    fic_t = all_fic_t  

    return idx_G,idx_LR, max_freq,max_power,corr_to_sc,homeostatic_fittness,rates,fic_t


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
    max_freq = results[2]
    max_power = results[3]    
    corr_to_sc = results[4]
    homeostatic_fittness = results[5]
    rates = results[6]
    fic_t = results[7]    
    max_freq_grid[:,idx_G,idx_LR] = max_freq
    max_power_grid[:,idx_G,idx_LR] = max_power        
    corr_to_sc_grid[:,idx_G,idx_LR] = corr_to_sc
    homeostatic_fittness_grid[:,idx_G,idx_LR] = homeostatic_fittness
    rates_grid[:,idx_G,idx_LR,:] = rates
    fic_t_grid[:,idx_G,idx_LR,:] = fic_t


    import os

# Assuming these arrays are already populated with data

arrays_to_save = {
    'max_freq_grid': max_freq_grid,
    'max_power_grid': max_power_grid,    
    'corr_to_sc_grid': corr_to_sc_grid,
    'homeostatic_fittness_grid': homeostatic_fittness_grid,
    'rates_grid': rates_grid,
    'fic_t_grid': fic_t_grid
}

results_folder = "./Results/slow_waves08"
# If the folder does not exist, create it
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    

# Save
for array_name, array_data in arrays_to_save.items():
    file_name = os.path.join(results_folder, f"{array_name}.npy")
    np.save(file_name, array_data)
