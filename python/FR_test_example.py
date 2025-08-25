import fastdyn_fic_dmf as dmf
import numpy as np
from scipy.io import loadmat
from scipy.stats import pearsonr
from multiprocessing import Pool
import os

# Parameters
nb_steps = 20000
SEED_REPS = 10
burn = 10000
C = loadmat('./data/DTI_fiber_consensus_HCP.mat')['connectivity'][:200, :200]
C = (0.2 * C / np.max(C)).astype(np.float32)
fit_res = np.load('./data/fit_res_3-44.npy')
b, a = fit_res[0].astype(np.float32), fit_res[1].astype(np.float32)
G_values = np.arange(0.1, 5, 0.1, dtype=np.float32)
NUM_CORES = 49
out_dir = './Results/ChequeoPreliminar'
os.makedirs(out_dir, exist_ok=True)
N = C.shape[0]



def single_dyn(idx):
    G = G_values[idx]
    base = dmf.default_params(C=C)
    decay = float(np.exp(a + np.log(10) * b))
    base.update({
        'G': float(G),
        'obj_rate': 3.44,
        'taoj_vector': np.ones(N)*decay,
        'lr_vector': np.ones(N)*10,
        'with_decay': True,
        'return_rate': True,
        'return_fic': True,
        'return_bold': False
    })
    base['J'] = (0.75 * G * base['C'].sum(axis=0).squeeze() + 1).astype(np.float32)
    fr_means = np.zeros(SEED_REPS, dtype=np.float32)
    fic_reps = np.zeros((SEED_REPS, N), dtype=np.float32)
    for rep in range(SEED_REPS):
        params = base.copy()
        params['seed'] = rep + 1
        params['with_plasticity'] = True
        rates, _, _, fic = dmf.run(params, nb_steps)
        fr_means[rep] = float(rates[:, burn:].mean())
        fic_reps[rep] = fic.mean(axis=1).astype(np.float32)
    return fr_means, fic_reps


print('Starting dynamic stage...')
with Pool(NUM_CORES) as pool:
    dyn_res = pool.map(single_dyn, range(len(G_values)))
fr_dyn_reps, fic_dyn_reps = zip(*dyn_res)
fr_dyn_reps = np.stack(fr_dyn_reps, axis=0).astype(np.float32)
fic_dyn_reps = np.stack(fic_dyn_reps, axis=0).astype(np.float32)  # (n_G, SEED_REPS, N)
print(fr_dyn_reps)
#np.save(os.path.join(out_dir, 'avg_fr_dyn_reps.npy'), fr_dyn_reps)
#np.save(os.path.join(out_dir, 'dyn_fic_reps.npy'), fic_dyn_reps)
print('Dynamic stage completed.')