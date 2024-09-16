import fastdyn_fic_dmf as dmf
import numpy as np
from scipy.io import loadmat
import os
from multiprocessing import Pool, Manager

C = loadmat('./data/SC_and_5ht2a_receptors.mat')['sc90']
C = 0.2 * C / np.max(C)
params = dmf.default_params(C=C)

# Main setup for this simulation
params["return_rate"] = True
params["return_bold"] = False
params["return_fic"] = True  # Ensure fic_t is returned
params["with_plasticity"] = True
params["with_decay"] = True

G_range = np.arange(0, 8.5, 0.5)
LR_range = np.logspace(0, 3, 100)
DECAY_range = np.logspace(2, 6, 110)


burnout = 5
nb_steps = 50000

# Define the number of cores to use
NUM_CORES = 24

mean_hom_fit_grid = np.zeros((len(DECAY_range), len(LR_range)))
std_hom_fit_grid = np.zeros((len(DECAY_range), len(LR_range)))
hom_grid = np.zeros((len(DECAY_range), len(LR_range), len(G_range)))
fic_t_grid = np.zeros((len(DECAY_range), len(LR_range), len(G_range), C.shape[0]))

def grid_step(args):
    DECAY_tuple, LR_tuple = args
    idx_LR, LR = LR_tuple[0], LR_tuple[1]
    idx_DECAY, DECAY = DECAY_tuple[0], DECAY_tuple[1]
    #print(f"Doing {DECAY} {LR}")
    OBJ_RATE = 3.44    
    params['lrj'] = LR
    params['taoj'] = DECAY 
    params['obj_rate'] = OBJ_RATE
    all_fits = []
    all_fic_t = np.zeros((len(G_range), C.shape[0]))
    for idx_G in range(len(G_range)):        
        params['G'] = G_range[idx_G]
        params['seed'] = idx_G
        params['J'] = 0.75 * params['G'] * params['C'].sum(axis=0).squeeze() + 1
        rates, _, _, fic_t = dmf.run(params, nb_steps)     
        
        rates = rates[:, np.ceil(burnout * 1000).astype(int):]
        all_fits.append(OBJ_RATE - np.mean(rates))
        all_fic_t[idx_G] = np.mean(fic_t, axis=1)
    
    mean_homeostatic_fittness = np.mean(np.array(all_fits))
    std_homeostatic_fittness = np.std(np.array(all_fits))
    
    return idx_DECAY, idx_LR, mean_homeostatic_fittness, std_homeostatic_fittness, np.array(all_fits), np.array(all_fic_t)

# Create a list of argument tuples for the nested loop function
args_list = [((idx_DECAY, DECAY), (idx_LR, LR))
             for idx_DECAY, DECAY in enumerate(DECAY_range)             
             for idx_LR, LR in enumerate(LR_range)]

manager = Manager()
results_list = manager.list()

# Create a pool of worker processes
with Pool(processes=NUM_CORES) as pool:
    # Map the nested loop function to the argument list across multiple processes
    results_list.extend(pool.map(grid_step, args_list))

for results in results_list:
    idx_DECAY = results[0]    
    idx_LR = results[1]
    mean_hom = results[2]
    std_hom = results[3] 
    hom = results[4]
    fic_t = results[5]
    
    mean_hom_fit_grid[idx_DECAY, idx_LR] = mean_hom 
    std_hom_fit_grid[idx_DECAY, idx_LR] = std_hom
    hom_grid[idx_DECAY, idx_LR] = hom
    fic_t_grid[idx_DECAY, idx_LR] = fic_t
    
arrays_to_save = {
    'mean_hom_fit_grid': mean_hom_fit_grid,
    'std_hom_fit_grid': std_hom_fit_grid,
    'hom_grid': hom_grid,
    'fic_t_grid': fic_t_grid        
}

results_folder = "./Results/Figure1/HomeostaticFit3-44-Grid"
if not os.path.exists(results_folder):
    # If not, create the folder
    os.makedirs(results_folder)
    print(f"Folder '{results_folder}' created successfully.")
else:
    print(f"Folder '{results_folder}' already exists.")
    
# Save
for array_name, array_data in arrays_to_save.items():
    file_name = os.path.join(results_folder, f"{array_name}.npy")
    np.save(file_name, array_data)
