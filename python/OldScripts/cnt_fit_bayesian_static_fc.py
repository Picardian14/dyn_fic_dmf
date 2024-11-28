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
import pickle as pkl
from skopt import gp_minimize
from joblib import Parallel, delayed
from skopt.plots import plot_convergence
#from mne.time_frequency import psd_array_multitaper

MODEL_TYPE = "static_fc"
RANDOM_STATE=1
folder_path = f"./Results/fit_coma_cnt_bayesian/{MODEL_TYPE}-{RANDOM_STATE}/"

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
params['N'] = len(C)
# Main setup for this simulation
params["return_rate"] = True
params["return_bold"] = True
params["return_fic"] = True
# These are now default true
params["with_plasticity"] = False
params["with_decay"] = False


burnout = 7
flp = 0.01
fhp = 0.1
b_filter,a_filter = butter(2,np.array([flp, fhp])*2*params['TR'], btype='band')
wsize = 30
overlap = 29
T = 250 - burnout
win_start = np.arange(0, T - wsize - 1, wsize - overlap)
nwins = len(win_start)
isubfc = np.triu_indices(C.shape[1],1)
nints = len(isubfc[0])
isubfcd = np.triu_indices(nwins,1)
NSUB = 13
emp_fcds = np.zeros((NSUB,nwins,nwins))
emp_fcs = np.zeros((NSUB,params['N'],params['N']))



for i in range(NSUB):
    bold = data[i]
    bold = bold[:, burnout:]    
    filt_bold = lfilter(b_filter,a_filter,bold)
    time_fc = compute_fcd(filt_bold.T, wsize, overlap, isubfc)    
    bold_fc = np.corrcoef(filt_bold)
    fcd = np.corrcoef(time_fc.T)       
    emp_fcds[i] = fcd
    emp_fcs[i] = bold_fc

fit_res = np.load("./data/fit_res_3-44.npy")
b = fit_res[0] # First element is the slope
a = fit_res[1]
params['N'] = C.shape[0]
params['TR'] = 2
nb_steps = int((data.shape[-1]-burnout)*params['TR']/params['dtt'])


# Define the number of cores to use
NUM_CORES = NSUB+1

def parallelized_function(rep, params, nb_steps, burnout, TR, b_filter, a_filter, wsize, overlap, isubfc):
    _, _, bold, _ = dmf.run(params, nb_steps)         
    filt_bold = lfilter(b_filter, a_filter, bold)
    time_fc = compute_fcd(filt_bold.T, wsize, overlap, isubfc)
    bold_fc = np.corrcoef(filt_bold)
    fcd = np.corrcoef(time_fc.T)
    return bold_fc, fcd

def run_parallel(NSUB, params, nb_steps, burnout, TR, b_filter, a_filter, wsize, overlap, isubfc):
    sim_fcs = Parallel(n_jobs=NSUB)(delayed(parallelized_function)(rep, params, nb_steps, burnout, TR, b_filter, a_filter, wsize, overlap, isubfc) for rep in range(NSUB))
    return sim_fcs


def dmf_step_dyn(args):
    sim_fcds = np.zeros((NSUB,nwins,nwins))
    sim_fcs = np.zeros((NSUB,params['N'],params['N']))
    G, ALPHA = args    
    print(f"Doing {G} {ALPHA}")    
    OBJ_RATE = 3.44        
    params['G'] = G
    # Using heuristic linear rule     
    params['obj_rate'] = OBJ_RATE    
    params['J'] = ALPHA*params['G']*params['C'].sum(axis=0).squeeze() + 1
    params['with_decay'] = False
    params['with_plasticity]'] = False
    params['seed']=RANDOM_STATE
    parallel_results = run_parallel(NSUB, params, nb_steps, burnout, params['TR'], b_filter, a_filter, wsize, overlap, isubfc)
    for rep in range(len(parallel_results)):
        sim_fcs[rep,:,:] = parallel_results[rep][0]
        sim_fcds[rep,:,:] = parallel_results[rep][1]
    
    crashed = np.isnan(np.mean(sim_fcs,axis=0)).any()
    mean_fc = np.mean(sim_fcs,axis=0) if not crashed else np.zeros((params['N'],params['N']))        

    corr_to_sc = 0 if crashed else pearsonr(mean_fc[isubfc[0],isubfc[1]], np.mean(emp_fcs,axis=0)[isubfc[0],isubfc[1]])[0]              
    ks, p = 1, 1 if crashed else ks_2samp(sim_fcds[:,isubfcd[0],isubfcd[1]].flatten(),emp_fcds[:,isubfcd[0],isubfcd[1]].flatten())        
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)    
    filename = f"mean-fc.txt"    
    file_path = os.path.join(folder_path, filename)    
    np.savetxt(file_path, mean_fc)

    filename = f"corr-to-sc.txt"    
    file_path = os.path.join(folder_path, filename)    
    with open(file_path, 'w') as file:
        file.write(str(corr_to_sc))

    filename = f"ks.txt"    
    file_path = os.path.join(folder_path, filename)    
    with open(file_path, 'w') as file:
        file.write(str(ks))
    
   
    return 1-corr_to_sc



def intermediate_save(res):
    
    # Load mean-fc.txt
    filename_mean_fc = "mean-fc.txt"
    file_path_mean_fc = os.path.join(folder_path, filename_mean_fc)
    mean_fc = np.loadtxt(file_path_mean_fc)
    # Load corr-to-sc.txt
    filename_corr_to_sc = "corr-to-sc.txt"
    file_path_corr_to_sc = os.path.join(folder_path, filename_corr_to_sc)
    with open(file_path_corr_to_sc, 'r') as file:
        corr_to_sc = float(file.read())
    
    file_path = os.path.join(folder_path, "best_result.txt")

    # Check if the file exists
    if os.path.exists(file_path):
        # If the file exists, load the previous best result
        with open(file_path, 'r') as file:
            best_result = float(file.read())

        # Compare with the current result and update if necessary
        if res.fun < best_result:
            with open(file_path, 'w') as file:
                file.write(str(res.fun))
            iter_folder_path = os.path.join(folder_path, f"iter_{len(res.x_iters)}")
            if not os.path.exists(iter_folder_path):
               os.makedirs(iter_folder_path)   
            filename = f"mean-fc.txt"    
            file_path = os.path.join(iter_folder_path, filename)    
            np.savetxt(file_path, mean_fc)

            filename = f"corr-to-sc.txt"    
            file_path = os.path.join(iter_folder_path, filename)    
            with open(file_path, 'w') as file:
                file.write(str(corr_to_sc))

            filename = f"res.pkl"    
            file_path = os.path.join(iter_folder_path, filename)    
            with open(file_path, 'wb') as file:
                pkl.dump(res,file)

    else:
        # If the file doesn't exist, create it and save the current result
        with open(file_path, 'w') as file:
            file.write(str(res.fun))
        iter_folder_path = os.path.join(folder_path, f"iter_{len(res.x_iters)}")
        if not os.path.exists(iter_folder_path):
            os.makedirs(iter_folder_path)   
        filename = f"mean-fc.txt"    
        file_path = os.path.join(iter_folder_path, filename)    
        np.savetxt(file_path, mean_fc)

        filename = f"corr-to-sc.txt"    
        file_path = os.path.join(iter_folder_path, filename)    
        with open(file_path, 'w') as file:
            file.write(str(corr_to_sc))

        filename = f"res.pkl"    
        file_path = os.path.join(iter_folder_path, filename)    
        with open(file_path, 'wb') as file:
            pkl.dump(res,file)



G_range = (0.0, 8.0)
ALPHA_range = (0.01,1.0)

res = gp_minimize(dmf_step_dyn,
                  [G_range,ALPHA_range],
                  acq_func="EI",      # the acquisition function
                  n_calls=200,         # the number of evaluations of f
                  n_random_starts=10,  # the number of random initialization points
                  noise=1e-10,       # the noise level (optional)
                  random_state=RANDOM_STATE,
                  callback=intermediate_save,
                  n_jobs=-1)


import pickle
res_file = os.path.join(folder_path, "res_object.pkl")
with open(res_file, 'wb') as file:
    pickle.dump(res, file)

plot_convergence(res)
plot_file = os.path.join(folder_path, "convergence.png")
plt.savefig(plot_file)