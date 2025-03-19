# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:36:44 2023

@author: rherz
# Clustering dream reports
"""
import numpy as np
import pandas as pd
import dclus as dclust
from matplotlib.pyplot import cm
from scipy.stats import zscore
from scipy.spatial import distance


#%% Loading csv
basefold = 'C:\\Users\\rherz\\OneDrive\\Documents\\Projects\\NicoClustering\\data\\' # change here for the folder where the csv is
filename = 'nap1_subj_dim_scores_clustering.csv' # name of the csv

datatable = pd.read_csv(basefold+filename)
data = datatable.to_numpy()
data = np.array(data[:,1:],dtype=float)

#%% Running PCA and clustering
numpcs = 3
ishalo = 0
dc = 0.08 # neighborhood parameter. EXPLORE WITH THIS. fOR VALUES <.04 will generate an error for this data
ci_alpha = 0.2 # threshold for centroids detection. EXPLORE THIS AS WELL
# PCA
distmat,proj_data = dclust.pca_and_distance(data.T,numpcs)
# Cluserting
rho,delta,centid,cluslabels,halolabels,threshold = dclust.clustering_by_density(distmat,dc,ci_alpha,ishalo)
nclus = np.max(cluslabels)
color = cm.tab20(np.linspace(0, 1, nclus))
dclust.plot_clustering_summary(delta,rho,centid,cluslabels,threshold,proj_data,color)

#%% The same but without PCA
norm_data = zscore(data,axis=0)
raw_distmat = distance.cdist(norm_data, norm_data, 'euclidean')

# Cluserting
dc = 0.08 # neighborhood parameter. EXPLORE WITH THIS. fOR VALUES <.04 will generate an error for this data
ci_alpha = 0.1 # threshold for centroids detection. EXPLORE THIS AS WELL
raw_rho,raw_delta,raw_centid,raw_cluslabels,raw_halolabels,raw_threshold = dclust.clustering_by_density(raw_distmat,dc,ci_alpha,ishalo)
raw_nclus = np.max(raw_cluslabels)
raw_color = cm.tab20(np.linspace(0, 1, raw_nclus))
dclust.plot_clustering_summary(raw_delta,raw_rho,raw_centid,raw_cluslabels,raw_threshold,proj_data,raw_color)
