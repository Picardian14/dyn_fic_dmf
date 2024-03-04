import fastdyn_fic_dmf as dmf
from scipy.stats import zscore, pearsonr
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
# Fetch default parameters
import tracemalloc
from scipy.io import loadmat
from scipy.stats import zscore
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

def grid_step(args):
    sampling_freq = 10000
    nb_steps = 100000
    C = loadmat('../SC_and_5ht2a_receptors.mat')['sc90']
    C = 0.2*C/np.max(C)
    triu_idx = np.triu_indices(C.shape[1],1)
    brunout = 5
    params = dmf.default_params(C=C)
    params['N'] = C.shape[0]
    G_tuple, DECAY_tuple, OBJ_RATE_tuple, LR_tuple = args
    idx_G,G = G_tuple[0],G_tuple[1]
    idx_DECAY,DECAY = DECAY_tuple[0],DECAY_tuple[1]
    idx_OBJ_RATE,OBJ_RATE = OBJ_RATE_tuple[0],OBJ_RATE_tuple[1]
    idx_LR,LR = LR_tuple[0],LR_tuple[1]
    print(f"Running - G:{G} / DECAY:{DECAY} / OBJ_RATE:{OBJ_RATE} / LR:{LR} \n")
    with_decay = DECAY>0
    params['lrj'] = LR
    params['G'] = G
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
    # Embedded array whose first elemnt is the index of the first number equal or greater then 4 or 1
    four_idx_freq = np.where(frequencies >= 4)[0][0]+1    
    one_idx_freq = np.where(frequencies >= 1)[0][0]
    plow_ptot = np.sum(power_spectrum[:, one_idx_freq:four_idx_freq], axis=1)/np.sum(power_spectrum[:,:], axis=1)                                        

    amount_slow_regions = np.sum(plow_ptot>0.3)
    mean_slow = np.mean(plow_ptot)
    std_slow = np.std(plow_ptot)
    corr_to_sc = pearsonr(rates_fc[triu_idx[0],triu_idx[1]], C[triu_idx[0],triu_idx[1]])[0]

    homeostatic_fittness =  OBJ_RATE - np.mean(rates)  
    return idx_G, idx_DECAY,idx_OBJ_RATE,idx_LR, amount_slow_regions,mean_slow, std_slow,corr_to_sc ,homeostatic_fittness,np.mean(rates,axis=1),np.mean(fic_t,axis=1)


def bayes_step(G, DECAY, LR):
    OBJ_RATE = 3.44
    sampling_freq = 10000
    nb_steps = 100000
    C = loadmat('../SC_and_5ht2a_receptors.mat')['sc90']
    C = 0.2*C/np.max(C)
    triu_idx = np.triu_indices(C.shape[1],1)
    brunout = 5
    params = dmf.default_params(C=C)
    params['N'] = C.shape[0]
    #print(f"Running - G:{G} / DECAY:{DECAY} / OBJ_RATE:{OBJ_RATE} / LR:{LR} \n")
    with_decay = DECAY>0
    params['lrj'] = LR
    params['G'] = G
    params['taoj'] = DECAY if with_decay else 10 # If 0 it means no plasticity at all. We put some value so it does not crash
    params['obj_rate'] = OBJ_RATE
    #params['taoj'] = 210000
    params['J'] = 0.75*params['G']*params['C'].sum(axis=0).squeeze() + 1
    rates, rates_inh, _, fic_t = dmf.run(params, nb_steps,
                                            return_rate=True, return_bold=False, return_fic=True, 
                                            with_decay=with_decay, with_plasticity=True)        
    rates = rates[:, np.ceil(brunout * 1000).astype(int):]
  
    homeostatic_fittness =  np.abs(OBJ_RATE - np.mean(rates))
    return homeostatic_fittness


def plot_and_calculate_general_plot(params,rates, rates_inh, bold, fic_t):
    brunout = 5
    # Slicing and computations
    bold = bold[:, (np.ceil(brunout / params['TR'])).astype(int):]
    rates = rates[:, np.ceil(brunout * 1000).astype(int):]
    rates_inh = rates_inh[:, np.ceil(brunout * 1000).astype(int):]
    fic_t = fic_t[:, np.ceil(brunout * 1000).astype(int):]
    rates_fc = np.corrcoef(rates)
    rates_inh_fc = np.corrcoef(rates_inh)
    T = len(bold.T)
    isubfcd = np.triu_indices(C.shape[1],1)
    flp = 0.01
    fhp = 0.1
    wsize = 30
    overlap = 29
    win_start = np.arange(0, T - wsize, wsize - overlap)
    nwins = len(win_start)
    nints = len(isubfcd[0])
    b,a = butter(2,np.array([flp, fhp])*2*params['TR'], btype='band')
    filt_bold = lfilter(b,a,bold)
    time_fc = compute_fcd(filt_bold.T, wsize, overlap, isubfcd)
    # Replace 'compute_fcd' with the appropriate function or code that computes time_fc
    bold_fc = np.corrcoef(filt_bold)
    fcd = np.corrcoef(time_fc)
    mean_fic_dist = np.zeros(N)
    for n in range(N):
        data = fic_t[:, n]
        mean_fic_dist[n] = np.mean(data)


    # Assuming 'rates' is your firing rates variable
    sampling_freq = 10000  # Replace with your actual sampling frequency (e.g., 1000 Hz)
    # Calculate the power spectral density using multitaper method
    power_spectrum,frequencies = psd_array_multitaper(rates, sfreq=sampling_freq, fmin=0, fmax=100,bandwidth=32*(sampling_freq/rates.shape[1]))

    fig = plt.figure(figsize=(15, 9))
    #plt.title(f"G: {G_range[obs_idx_G]} -- LR: {LR_range[obs_idx_LR]}")
    plt.title(f"G: {params['G']} -- LR: {params['lrj'] if with_plasticity else 'No plasticity '} -- Decay:{params['taoj'] if with_decay else 'No decay'}")
    sel_t = np.arange(1, rates.shape[1] + 1)
    sel_t_bold = np.arange(1, bold.shape[1] + 1)
    plt.subplot(3, 1, 1)
    plt.plot(sel_t * 0.001, np.mean(rates_inh, axis=0), 'b')
    plt.plot(sel_t * 0.001, np.mean(rates, axis=0), 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('E Firing Rates (Hz)')
    plt.twinx()
    plt.plot(sel_t * 0.001, np.mean(fic_t, axis=0), color=[0, 0.5, 0])
    plt.ylabel('FIC Y-axis')


    plt.subplot(3, 1,2)
    plt.plot(sel_t_bold, filt_bold.T)
    plt.xticks((sel_t_bold - 1) * params['TR'])
    plt.xlim([0, sel_t_bold[-1]])
    plt.xlabel('Time (s)')
    plt.ylabel('BOLD(Hz)')

    plt.subplot(3, 5, 11)
    plt.bar(np.arange(params['N']), np.mean(rates, axis=1), edgecolor='none')
    plt.plot([0, params['N'] + 1], [params['obj_rate'], params['obj_rate']], 'r--')
    plt.ylabel('E Firing Rate (Hz)')
    plt.xlabel('Regions')

    plt.subplot(3, 5, 12)
    plt.imshow(rates_fc - np.eye(params['N']))
    plt.title('FC Rates')
    plt.colorbar()

    plt.subplot(3, 5, 13)
    # Assuming 'bold_fc' is defined
    plt.imshow(bold_fc - np.eye(params['N']))
    plt.title('FC BOLD')
    plt.colorbar()

    plt.subplot(3, 5, 14)
    plt.semilogy(frequencies, power_spectrum[0,:])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title(f'PSD Plow/Ptot: {np.sum(power_spectrum[0, :39])/np.sum(power_spectrum[0,:]):.2f}')

    plt.subplot(3, 5, 15)
    plt.imshow(fcd)
    plt.title('FCD')
    plt.colorbar()

    plt.tight_layout()
    plt.show()