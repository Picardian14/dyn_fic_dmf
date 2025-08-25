import numpy as np
from scipy.signal import butter, filtfilt, detrend
from scipy.stats import zscore

def filter_bold(bold, flp, fhp, tr):
    
    T, N = bold.shape
    fnq = 1 / (2 * tr)  # Nyquist frequency
    Wn = [flp / fnq, fhp / fnq]  # Butterworth bandpass non-dimensional frequency
    k = 2  # 2nd order Butterworth filter
    bfilt, afilt = butter(k, Wn, btype='band')  # Construct the filter

    # Filtering and plotting
    filt_bold = np.zeros((T, N))
    nzeros = 40
    aux_filt = detrend(bold, axis=0)
    aux_filt = np.concatenate((np.zeros((nzeros, N)), aux_filt, np.zeros((nzeros, N))))

    for n in range(N):
        aux_filt2 = filtfilt(bfilt, afilt, aux_filt[:, n])  # Zero-phase filter the data
        filt_bold[:, n] = zscore(aux_filt2[nzeros:-nzeros])

    return filt_bold

from nilearn import datasets, plotting, image
import numpy as np
def print_pet():
        
    # Load PET scan NIfTI image
    pet_img = image.load_img('/mnt/data2/home/BACKUP/hansen_receptors/data/PET_nifti_images/NMDA_ge179_hc29_galovic.nii.gz')

    # Normalize the PET image based on actual data min-max
    data = pet_img.get_fdata()
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Create a normalized Nifti image
    pet_img_norm = image.new_img_like(pet_img, data_norm)

    # Smooth for better visualization
    pet_img_smooth = image.smooth_img(pet_img_norm, fwhm=6)

    # Load fsaverage surface mesh
    fsaverage = datasets.fetch_surf_fsaverage()

    # Explicitly define threshold to exclude very low intensities clearly
    threshold = 0.1  # Adjust if needed for clarity

    # Plot with 'hot' colormap for smoother continuous gradients
    plotting.plot_img_on_surf(
        pet_img_smooth,
        views=['lateral', 'medial'],
        hemispheres=['left'],  # single hemisphere
        threshold=threshold,
        cmap='hot',  # Smooth continuous gradient
        colorbar=True,
        bg_on_data=True,  # grey background for values below threshold
        vmax=1.0,        # ensures full color range usage
        surf_mesh='fsaverage',
        title='PET Scan Cortical Projection (Normalized, Hot cmap)'
    )
