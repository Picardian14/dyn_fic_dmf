import fastdyn_fic_dmf as dmf
import numpy as np
import matplotlib.pyplot as plt
# Fetch default parameters
from scipy.io import loadmat
from scipy.stats import zscore, pearsonr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore, pearsonr
from multiprocessing import Pool
# Helper functions

nb_steps = 50000
C = loadmat('./data/DTI_fiber_consensus_HCP.mat')['connectivity'][:200, :200]
C = 0.2*C/np.max(C)
fit_res = np.load("./data/fit_res_3-44.npy")
b = fit_res[0] # First element is the slope
a = fit_res[1]
SEED_REPS = 30

def get_stat_dyn_fic(G_val):
    params = dmf.default_params(C=C)
    params['seed'] = 1
    params['G'] = G_val
    params['obj_rate'] = 3.44
    LR = 10
    DECAY = np.exp(a+np.log(LR)*b)
    params['taoj'] = DECAY
    params['lrj'] = LR
    params['J'] = 0.75*params['G']*params['C'].sum(axis=0).squeeze() + 1
    N = C.shape[0]
    params["with_decay"] = True
    params["with_plasticity"] = True
    params["return_fic"] = True
    params['return_rate'] = True
    params['return_fic'] = True
    params['return_bold'] = False
    rates_dyn, _, _, fic_t_dyn = dmf.run(params, nb_steps)
    params["with_plasticity"] = False
    rates_stat, _, _, fic_t_stat = dmf.run(params, nb_steps)
    burnout = 10000

    dyn_fic = np.mean(fic_t_dyn, axis=1)
    stat_fic = params['J']
    params['J'] = dyn_fic
    all_rates_mixed = np.zeros((SEED_REPS, N, nb_steps))
    for rep in range(SEED_REPS):
        params['seed'] = rep + 1 # Change the seed for each repetition add 1 to avoid seed 0
        rates_mixed, _, _, _ = dmf.run(params, nb_steps)
        all_rates_mixed[rep] = rates_mixed
    # Calculate the MSE between the dynamic and static FIC
    mse = np.mean((dyn_fic - stat_fic)**2)
    # calcualte the correlation between the dynamic and static FIC
    corr = pearsonr(dyn_fic, stat_fic)[0]

    # Calcualte the average excitatory firing rate across the time 
    # steps for the dynamic and static simulations
    avg_fr_dyn = np.mean(rates_dyn[:,burnout:], axis=1)
    avg_fr_stat = np.mean(rates_stat[:,burnout:], axis=1)
    avg_fr_mixed = np.mean(all_rates_mixed[:,:,burnout:], axis=2)
    return mse,corr, avg_fr_dyn, avg_fr_stat,avg_fr_mixed, stat_fic, dyn_fic 

def parallel_function(G):
    return get_stat_dyn_fic(G)


# Define the range of G values from 0.1 to 5 with a step of 0.1
G_values = np.arange(0.1, 5, 0.1)
mse_values = np.zeros(len(G_values))
corr_values = np.zeros(len(G_values))
avg_fr_dyn_values = np.zeros((len(G_values), C.shape[0]))
avg_fr_stat_values = np.zeros((len(G_values), C.shape[0]))
avg_fr_mixed_values = np.zeros((SEED_REPS,len(G_values), C.shape[0]))
stat_fic_values = np.zeros((len(G_values), C.shape[0]))
dyn_fic_values = np.zeros((len(G_values), C.shape[0]))

NUM_CORES = 16
with Pool(processes=NUM_CORES) as pool:
        results = pool.map(parallel_function, G_values)

for i, result in enumerate(results):
    mse, corr, avg_fr_dyn, avg_fr_stat, avg_fr_mixed, stat_fic, dyn_fic = result
    mse_values[i] = mse
    corr_values[i] = corr
    avg_fr_dyn_values[i, :] = avg_fr_dyn
    avg_fr_stat_values[i, :] = avg_fr_stat
    avg_fr_mixed_values[:,i, :] = avg_fr_mixed
    stat_fic_values[i] = stat_fic
    dyn_fic_values[i] = dyn_fic
    

# Save the results in ~/Desktop/DatosParaRuben/ChequeoPreliminar as separate .npy files
data_path = "~/Desktop/DatosParaRuben/ChequeoPreliminar"
np.save("./Results/ChequeoPreliminar/mse_values.npy", mse_values)
np.save("./Results/ChequeoPreliminar/corr_values.npy", corr_values)
np.save("./Results/ChequeoPreliminar/avg_fr_dyn_values.npy", avg_fr_dyn_values)
np.save("./Results/ChequeoPreliminar/avg_fr_stat_values.npy", avg_fr_stat_values)
np.save("./Results/ChequeoPreliminar/avg_fr_mixed_values.npy", avg_fr_mixed_values)
np.save("./Results/ChequeoPreliminar/stat_fic_values.npy", stat_fic_values)
np.save("./Results/ChequeoPreliminar/dyn_fic_values.npy", dyn_fic_values)
np.save("./Results/ChequeoPreliminar/G_values.npy", G_values)

