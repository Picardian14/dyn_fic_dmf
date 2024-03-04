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

C = loadmat('data/SC_and_5ht2a_receptors.mat')['sc90']
C = 0.2*C/np.max(C)
params = dmf.default_params(C=C)

# Main setup for this simulation
params["return_rate"] = True
params["return_bold"] = True
params["return_fic"] = True
# These are now default true
params["with_plasticity"] = False
params["with_decay"] = False

isubfcd = np.triu_indices(C.shape[1],1)
burnout = 5
flp = 0.01
fhp = 0.1
wsize = 30
overlap = 29
T = 192
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

# Load coefficients to estimte Decay with ALPHA

coeffs = loadmat('./data/LinearFitCoefficients.mat')
a = coeffs['a'][0][0]
b = coeffs['b'][0][0]
triu_idx = np.triu_indices(C.shape[1],1)
params['N'] = C.shape[0]

G_range = [1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5,2.6,2.7]
ALPHA_range = [0.6,0.65,0.7,0.75,0.8,0.]#np.arange(1,51,2)



params['TR'] = 2.4


nb_steps = int((data.shape[-1]-2*burnout)*params['TR']/params['dtt'])

NUM_CORES = 24
G_range = [1,2,3,4,5,6]
ALPHA_range = [10,200,300,1000]


fit_fc_grid = np.zeros((len(G_range),len(ALPHA_range)))
fit_fcd_grid = np.zeros((len(G_range),len(ALPHA_range)))
fc_grid = np.zeros((len(G_range),len(ALPHA_range), params['N'],params['N']))
fcd_grid = np.zeros((len(G_range),len(ALPHA_range), len(isubfcd[0]),len(isubfcd[0])))

burnout = 5



def grid_step(args):
    sim_fcds = np.zeros((NSUB,4005,4005))
    sim_fcs = np.zeros((NSUB,90,90))
    G_tuple, ALPHA_tuple = args
    idx_ALPHA,ALPHA = ALPHA_tuple[0],ALPHA_tuple[1]
    idx_G,G = G_tuple[0],G_tuple[1]
    print(f"Doing {G} {ALPHA}")
   
    OBJ_RATE = 3.44    
    params['G'] = G
    params['obj_rate'] = OBJ_RATE
    #params['taoj'] = 210000
    params['J'] = 0.75*params['G']*params['C'].sum(axis=0).squeeze() + 1
    for rep in range(NSUB):
        
        rates, rates_inh, bold, fic_t = dmf.run(params, nb_steps)     
        bold = bold[:, (np.ceil(burnout / params['TR'])).astype(int):]
        filt_bold = lfilter(b_filter,a_filter,bold)
        time_fc = compute_fcd(filt_bold.T, wsize, overlap, isubfcd)
        # Replace 'compute_fcd' with the appropriate function or code that computes time_fc
        bold_fc = np.corrcoef(filt_bold)
        fcd = np.corrcoef(time_fc)
        sim_fcs[rep] = bold_fc
        sim_fcds[rep] = fcd
    print("Finished reptitions")
    mean_fc = np.mean(sim_fcs,axis=0)
    corr_to_sc = pearsonr(mean_fc[triu_idx[0],triu_idx[1]], C[triu_idx[0],triu_idx[1]])[0]        
    print("Calcualted corr to FC")
    mean_fcds = np.mean(sim_fcds,axis=0)
    ks, p = ks_2samp(mean_fcds.flatten(),np.mean(emp_fcds,axis=0).flatten())
    print("Calcualted sim fcd")    
    folder_path = f"./Results/fit_coma_cnt/static/{idx_G}_{idx_ALPHA}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)    
    filename = f"mean-fc.txt"    
    file_path = os.path.join(folder_path, filename)    
    np.savetxt(file_path, mean_fc)

    #filename = f"mean-fcds.txt"    
    #file_path = os.path.join(folder_path, filename)    
    #np.savetxt(file_path, mean_fcds)

    filename = f"corr-to-sc.txt"    
    file_path = os.path.join(folder_path, filename)    
    with open(file_path, 'w') as file:
        file.write(str(corr_to_sc))

    filename = f"ks.txt"    
    file_path = os.path.join(folder_path, filename)    
    with open(file_path, 'w') as file:
        file.write(str(ks))

    return idx_G,idx_ALPHA, corr_to_sc,ks, mean_fc, mean_fcds


from multiprocessing import Pool,Manager


# Define the number of cores to use

# Create a list of argument tuples for the nested loop function
args_list = [((idx_G,G), (idx_ALPHA,ALPHA))
             for idx_G,G in enumerate(G_range)             
             for idx_ALPHA,ALPHA in enumerate(ALPHA_range)]

manager = Manager()
results_list = manager.list()
# Create a pool of worker processes
with Pool(processes=NUM_CORES) as pool:
    # Map the nested loop function to the argument list across multiple processes
    results_list.extend(pool.map(grid_step, args_list))


for results in results_list:
    idx_G = results[0]    
    idx_ALPHA = results[1]
    fit_fc = results[2]
    fit_fcd = results[3] 
    fc = results[4]  
    fcd = results[5]
    
    fit_fc_grid[idx_G,idx_ALPHA] = fit_fc
    fit_fcd_grid[idx_G,idx_ALPHA] = fit_fcd
    fc_grid[idx_G,idx_ALPHA] = fc
    fcd_grid[idx_G,idx_ALPHA] = fcd


import os

# Assuming these arrays are already populated with data

arrays_to_save = {
    'fit_fc_grid': fit_fc_grid,
    'fit_fcd_grid': fit_fcd_grid,
    'fc_grid': fc_grid,
    'fcd_grid': fcd_grid
    
}

results_folder = "./Results/fit_coma_cnt/static"

# Save
for array_name, array_data in arrays_to_save.items():
    file_name = os.path.join(results_folder, f"{array_name}.npy")
    np.save(file_name, array_data)
