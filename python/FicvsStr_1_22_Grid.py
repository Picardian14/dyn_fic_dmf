import fastdyn_fic_dmf as dmf
import numpy as np
from scipy.io import loadmat
import os
from multiprocessing import Pool, Manager

C = loadmat('./data/DTI_fiber_consensus_HCP.mat')['connectivity'][:200, :200]
C = 0.2 * C / np.max(C)
params = dmf.default_params(C=C)

# Main simulation setup
params["return_rate"] = True
params["return_bold"] = False
params["return_fic"] = True  # Ensure fic_t is returned
params["with_plasticity"] = True
params["with_decay"] = True

G_range = np.arange(0, 8.5, 0.5)
LR_range = np.logspace(0, 3, 100)
DECAY_range = np.logspace(2, 6, 110)

burnout = 7
nb_steps = 50000
NUM_CORES = 24
N_RANDOMIZATIONS = 8

node_strength = C.sum(axis=0)
# Number of time steps after the burnout period
post_burnout_steps = nb_steps - burnout * 1000

# We now store a time series of correlations for each parameter combination
# Dimensions: (DECAY, LR, randomization, G, time)
fic_cor_timeseries_grid = np.zeros(
    (len(DECAY_range), len(LR_range), N_RANDOMIZATIONS, len(G_range), post_burnout_steps), dtype=np.float32
)

def grid_step(args):
    (idx_DECAY, DECAY), (idx_LR, LR) = args
    print(f"Starting grid step: DECAY={DECAY}, LR={LR}")
    OBJ_RATE = 1.22
    params['lrj'] = LR
    params['taoj'] = DECAY
    params['obj_rate'] = OBJ_RATE

    # Prepare arrays for storing results of this DECAY/LR pair
    # Shape: (N_RANDOMIZATIONS, len(G_range), post_burnout_steps)
    all_random_fic_cor = np.zeros((N_RANDOMIZATIONS, len(G_range), post_burnout_steps), dtype=np.float32)

    for r in range(N_RANDOMIZATIONS):
        for idx_G, G_val in enumerate(G_range):
            params['G'] = G_val
            params['seed'] = idx_G + 1000*r
            reference_fic = 0.75 * params['G'] * params['C'].sum(axis=0).squeeze() + 1
            # create random vector that is in the same range as the reference fic
            params['J'] = np.random.rand(params['C'].shape[0]) * reference_fic.max()

            rates, _, _, fic_t = dmf.run(params, nb_steps)

            # Extract the post-burnout part of the FIC timeseries
            # fic_t shape: (nodes, time)
            fic_t_post_burnout = fic_t[:, np.ceil(burnout * 1000).astype(int):]

            # Compute correlation at each time step
            # For each time t, we correlate fic_t_post_burnout[:, t] with node_strength
            for t in range(post_burnout_steps):
                # Compute correlation with node_strength
                cor = np.corrcoef(fic_t_post_burnout[:, t], node_strength)[0, 1]
                all_random_fic_cor[r, idx_G, t] = cor

    return idx_DECAY, idx_LR, all_random_fic_cor

# Create a list of argument tuples for the nested loop function
args_list = [((idx_DECAY, DECAY), (idx_LR, LR))
             for idx_DECAY, DECAY in enumerate(DECAY_range)
             for idx_LR, LR in enumerate(LR_range)]

manager = Manager()
results_list = manager.list()

# Create a pool of worker processes
with Pool(processes=NUM_CORES) as pool:
    # Map the function to the argument list
    results_list.extend(pool.map(grid_step, args_list))

# Process results
for results in results_list:
    idx_DECAY = results[0]
    idx_LR = results[1]
    all_random_fic_cor = results[2]  # shape (N_RANDOMIZATIONS, len(G_range), post_burnout_steps)

    fic_cor_timeseries_grid[idx_DECAY, idx_LR] = all_random_fic_cor

arrays_to_save = {
    'fic_cor_timeseries_grid': fic_cor_timeseries_grid
}

results_folder = "./Results/Figure1/FicvsStr1-22-Grid"
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

print("Finished saving data.")
