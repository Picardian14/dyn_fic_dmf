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
from mne.time_frequency import psd_array_multitaper


def compute_fcd(data, wsize, overlap, isubdiag):
    T, N = data.shape
    win_start = np.arange(0, T - wsize - 1, wsize - overlap)
    nwins = len(win_start)
    fcd = np.zeros((len(isubdiag[0]), nwins))
    print(fcd.shape)
    print(data.shape)
    print((data[win_start[2]:win_start[2] + wsize + 1, :]).shape)
    for i in range(nwins):
        tmp = data[win_start[i]:win_start[i] + wsize + 1, :]
        cormat = np.corrcoef(tmp.T)
        fcd[:, i] = cormat[isubdiag[0],isubdiag[1]]
    return fcd


nb_steps = 100000
C = loadmat('../SC_and_5ht2a_receptors.mat')['sc90']
# Load coefficients to estimte Decay with LR
coeffs = loadmat('./data/LinearFitCoefficients.mat')
a = coeffs['a'][0][0]
b = coeffs['b'][0][0]
C = 0.2*C/np.max(C)
triu_idx = np.triu_indices(C.shape[1],1)
brunout = 5
params = dmf.default_params(C=C)
params['N'] = C.shape[0]
params['seed'] = 2
sampling_freq = 10000
G_max = 25
DECAY_max = 60000
OBJ_RATE_max = 15
G_step = 0.5
DECAY_step = 20000
OBJ_RATE_step = 3
G_range = np.arange(0,G_max,G_step)
LR_range = np.logspace(0, 3,100)

#G_range = [1,2]
#LR_range = [10,200]
# Define the number of cores to use
NUM_CORES = 16
peak_autocorrelation_grid = np.zeros((len(G_range),len(LR_range)))
peak_time_grid = np.zeros((len(G_range),len(LR_range)))
power_grid = np.zeros((len(G_range),len(LR_range), params['N'],951)) # Hardcoded number of frequencies with the given parameters of filtering
std_slow_grid = np.zeros((len(G_range),len(LR_range)))
corr_to_sc_grid = np.zeros((len(G_range),len(LR_range)))
homeostatic_fittness_grid = np.zeros((len(G_range),len(LR_range)))
rates_grid = np.zeros((len(G_range),len(LR_range),params['N']))
fic_t_grid = np.zeros((len(G_range),len(LR_range),params['N']))

from scipy.signal import find_peaks
def get_peak_and_ms(rates):
    """Get the value of the first peak from the autocorrealtion of the average firing rates"""
    signal = np.mean(rates, axis=0)
    signal -= np.mean(signal)

    # Calculate autocorrelation function
    autocorr = np.correlate(signal, signal, mode='full')

    # Normalize the autocorrelation function
    autocorr = autocorr / np.var(signal) / len(signal)
    autocorr = autocorr[len(signal)-1:]
    peaks, _ = find_peaks(autocorr, height=(0.2,0.8), prominence=0.2)
    if peaks.size==0:
        autocorr_value = 0
        time = 0
    else:
        autocorr_value = autocorr[peaks[0]]
        time = peaks[0]
    return autocorr_value,time


def grid_step(args):
    G_tuple, LR_tuple = args
    idx_LR,LR = LR_tuple[0],LR_tuple[1]
    idx_G,G = G_tuple[0],G_tuple[1]
    DECAY = np.exp(a+np.log(LR)*b)
    OBJ_RATE = 3.44
    print(f"Running - G:{G} / DECAY:{DECAY} / OBJ_RATE:{OBJ_RATE} / LR:{LR} \n")
    with_decay = DECAY>0
    params['lrj'] = LR
    params['G'] = G
    # Using heuristic linear rule 
    params['taoj'] = DECAY if with_decay else 10 # If 0 it means no plasticity at all. We put some value so it does not crash
    params['obj_rate'] = OBJ_RATE
    #params['taoj'] = 210000
    params['J'] = 0.75*params['G']*params['C'].sum(axis=0).squeeze() + 1
    rates, rates_inh, _, fic_t = dmf.run(params, nb_steps,
                                            return_rate=True, return_bold=False, return_fic=True, 
                                            with_decay=with_decay, with_plasticity=True)        
    rates = rates[:, np.ceil(brunout * 1000).astype(int):]
    rates_inh = rates_inh[:, np.ceil(brunout * 1000).astype(int):]
    fic_t = fic_t[:, np.ceil(brunout * 1000).astype(int):]
    rates_fc = np.corrcoef(rates)
    power_spectrum,frequencies = psd_array_multitaper(rates, verbose=False,sfreq=sampling_freq, fmin=0, fmax=100,bandwidth=32*(sampling_freq/rates.shape[1]))
    peak_autocorrelation,peak_time = get_peak_and_ms(rates)
    corr_to_sc = pearsonr(rates_fc[triu_idx[0],triu_idx[1]], C[triu_idx[0],triu_idx[1]])[0]
    
    homeostatic_fittness =  OBJ_RATE - np.mean(rates)  
    return idx_G,idx_LR, peak_autocorrelation,peak_time,corr_to_sc ,homeostatic_fittness,np.mean(rates,axis=1),np.mean(fic_t,axis=1), power_spectrum



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
    peak_autocorrelation = results[2]
    peak_time = results[3]    
    corr_to_sc = results[4]
    homeostatic_fittness = results[5]
    rates = results[6]
    fic_t = results[7]
    power_spectrum = results[8]
    peak_autocorrelation_grid[idx_G,idx_LR] = peak_autocorrelation
    peak_time_grid[idx_G,idx_LR] = peak_time    
    power_grid[idx_G, idx_LR] = power_spectrum
    corr_to_sc_grid[idx_G,idx_LR] = corr_to_sc
    homeostatic_fittness_grid[idx_G,idx_LR] = homeostatic_fittness
    rates_grid[idx_G,idx_LR,:] = rates
    fic_t_grid[idx_G,idx_LR,:] = fic_t


    import os

# Assuming these arrays are already populated with data

arrays_to_save = {
    'peak_autocorrelation_grid': peak_autocorrelation_grid,
    'peak_time_grid': peak_time_grid,
    'power_grid': power_grid,
    'corr_to_sc_grid': corr_to_sc_grid,
    'homeostatic_fittness_grid': homeostatic_fittness_grid,
    'rates_grid': rates_grid,
    'fic_t_grid': fic_t_grid
}

results_folder = "./Results/G_LR"

# Save
for array_name, array_data in arrays_to_save.items():
    file_name = os.path.join(results_folder, f"{array_name}.npy")
    np.save(file_name, array_data)


# Create subplots
from matplotlib.colors import LogNorm

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Heatmap for peak_time_grid
im1 = axes[0].imshow(peak_time_grid, cmap='viridis', origin='lower', aspect='auto', vmax=2000)
axes[0].set_title('Heatmap for peak_time_grid')
axes[0].set_xlabel('LR')
axes[0].set_ylabel('G')
# Set logarithmic tick labels for LR
log_labels = ['' for _ in range(len(LR_range))]
log_labels[0] = 1
log_labels[33] = 10
log_labels[66] = 100
log_labels[99] = 1000
axes[0].set_xticks(range(len(LR_range)))
axes[0].set_xticklabels(log_labels)
axes[0].set_yticks(range(0,50,5))
axes[0].set_yticklabels(G_range[range(0,50,5)])
plt.colorbar(im1, ax=axes[0], label='Values')
# Heatmap for peak_autocorrelation_grid
im2 = axes[1].imshow(peak_autocorrelation_grid, cmap='viridis', origin='lower', aspect='auto')
axes[1].set_title('Heatmap for peak_autocorrelation_grid')
axes[1].set_xlabel('LR')
axes[1].set_ylabel('G')
axes[1].set_xticks(range(len(LR_range)))
axes[1].set_xticklabels(log_labels)
axes[1].set_yticks(range(0,50,5))
axes[1].set_yticklabels(G_range[range(0,50,5)])
plt.colorbar(im2, ax=axes[1], label='Values')
plt.tight_layout()
plt.show()