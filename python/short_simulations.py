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
import seaborn as sns
import mat73

C = loadmat('./data/DTI_fiber_consensus_HCP.mat')['connectivity'][:200, :200]
C = 0.2*C/np.max(C)


#### FIRST EXAMPLE ####


G_val = 1.5

params = dmf.default_params(C=C)
fit_res = np.load("./data/fit_res_3-44.npy")
b = fit_res[0] # First element is the slope
a = fit_res[1]
nb_steps = 50000
params['seed'] = 1
params['G'] = G_val
params['obj_rate'] = 3.44
LR = 1
DECAY = np.exp(a+np.log(LR)*b)
params['taoj'] =  DECAY
params['lrj'] = LR
params['J'] = 0.75*params['G']*params['C'].sum(axis=0).squeeze() + 1
N = C.shape[0]
params["with_decay"] = True
params["with_plasticity"] = True
params['return_bold'] = False
params["return_fic"] = True
params["return_rate"] = True
rates_dyn, rates_inh_dyn, _, fic_t_dyn = dmf.run(params, nb_steps)
# save 
datafolder = '/home/ivan.mindlin/Desktop/DatosParaRuben/'
np.save(datafolder+'ChequeoPreliminar/rates_dyn.npy', rates_dyn)
np.save(datafolder+'ChequeoPreliminar/rates_inh_dyn.npy', rates_inh_dyn)
np.save(datafolder+'ChequeoPreliminar/fic_t_dyn.npy', fic_t_dyn)

#### FOR SLOW WAVES FIGURE ####

N=C.shape[0]
params = dmf.default_params(C=C)
fit_res = np.load("./data/fit_res_3-44.npy")
b = fit_res[0] # First element is the slope
a = fit_res[1]
nb_steps = 17000
burnout = 7
SEED = 1
params['seed'] = SEED
params['obj_rate'] = 3.44
LR = 100
params['lr_vector'] = np.ones(N)*LR
DECAY = np.exp(a+np.log(LR)*b)
params['taoj_vector'] =  DECAY
N = C.shape[0]
params["with_decay"] = True
params["with_plasticity"] = True
params["return_fic"] = True
params["return_rate"] = True

G_vals = [1.5,3,6]
e_rates_dyn = np.zeros((len(G_vals),N,nb_steps-burnout*1000))
inh_rates_dyn = np.zeros((len(G_vals),N,nb_steps-burnout*1000))
fic_dyn = np.zeros((len(G_vals),N,nb_steps-burnout*1000))

for G_val in G_vals:
    params['G'] = G_val
    params['J'] = 0.75*params['G']*params['C'].sum(axis=0).squeeze() + 1
    rates_dyn, rates_inh_dyn, _, fic_t_dyn = dmf.run(params, nb_steps)
    rates_dyn = rates_dyn[:,burnout*1000:]
    rates_inh_dyn = rates_inh_dyn[:,burnout*1000:]
    fic_t_dyn = fic_t_dyn[:,burnout*1000:]
    e_rates_dyn[G_vals.index(G_val),:,:] = rates_dyn
    inh_rates_dyn[G_vals.index(G_val),:,:] = rates_inh_dyn
    fic_dyn[G_vals.index(G_val),:,:] = fic_t_dyn

# save the results
datafolder = '/home/ivan.mindlin/Desktop/DatosParaRuben/'
np.save(datafolder+'slow_waves/time_series_examples/rates_results.npy', e_rates_dyn)
np.save(datafolder+'slow_waves/time_series_examples/fic_t_results.npy', fic_dyn)
np.save(datafolder+'slow_waves/time_series_examples/G_vals.npy', G_vals)

##### FOR CHIMERAS #####

# Load and normalize structural connectivity
C = loadmat('./data/DTI_fiber_consensus_HCP.mat')['connectivity'][:200, :200]
C = 0.2 * C / np.max(C)
N = C.shape[0]

# Get default parameters and fit coefficients
params = dmf.default_params(C=C)
fit_res = np.load("./data/fit_res_3-44.npy")
b = fit_res[0]  # slope
a = fit_res[1]

nb_steps = 17000
burnout = 7
SEED = 1
params['seed'] = SEED
params['obj_rate'] = 3.44

# Fix G to 3.5 and update J accordingly
params['G'] = 3.5
params['J'] = 0.75 * params['G'] * params['C'].sum(axis=0).squeeze() + 1

# Define two LR values
LR_list = [100, 10000]

# Prepare containers to store results for each LR value
e_rates_dyn_list = []
inh_rates_dyn_list = []
fic_dyn_list = []
LR_values = []

for LR in LR_list:
    params['lr_vector'] = np.ones(N) * LR
    DECAY = np.exp(a + np.log(LR) * b)
    params['taoj_vector'] = DECAY
    
    # Run the simulation
    rates_dyn, rates_inh_dyn, _, fic_t_dyn = dmf.run(params, nb_steps)
    
    # Remove the preliminary "burnout" time steps
    rates_dyn = rates_dyn[:, burnout * 1000:]
    rates_inh_dyn = rates_inh_dyn[:, burnout * 1000:]
    fic_t_dyn = fic_t_dyn[:, burnout * 1000:]
    
    e_rates_dyn_list.append(rates_dyn)
    inh_rates_dyn_list.append(rates_inh_dyn)
    fic_dyn_list.append(fic_t_dyn)
    LR_values.append(LR)

# Save the results
datafolder = '/home/ivan.mindlin/Desktop/DatosParaRuben/'
np.save(datafolder + 'chimeras/example_rates.npy', np.array(e_rates_dyn_list))