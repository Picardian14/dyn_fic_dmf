# Rubén Herzog 2023

import numpy as np
import scipy.stats
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.decomposition import PCA
# import py_mv_gc as mvit
import warnings
import random
import mne
import time
from scipy.stats import zscore
from scipy.io import loadmat




def clustering_by_density(dist,dc,alpha,ishalo):
    """
    # Searches for clusters on the 'dist' distance matrix using the density
    # peaks clustering methods by Rodriguez and Laio 2014.
    #
    # INPUTS
    #
    # 'dist' is a nbins x nbins matrix of distances between data bins
    # 'dc' average percentage of neighbours, ranging from [0, 1]
    # 'clusmet' is the method used for centroid search
    # 'alpha' is the percentile to compute prediction bounds on the centroid detection
    # 'ishalo' is 1 if halo cluster refinement is used and 0 otherwise
    
    #
    # OUTPUTS
    # 'rho' is the local density of point i
    # 'delta' is the minimum distance between point it and any other point with
    # higher density
    # 'centid' id of the cliuster centroid
    # 'cluslabels' vector with the cluster of each data point
    # 'halolabels' same than 'cluslabels' after halo refinement. 0 for non assigned points
    # 'threshold' threshold curve used to detect centroids (a function of alpha)

    """
    npts = len(dist)
    dist = dist*(1-np.eye(npts)) # removing diagonal from distance matrix
    rho = compute_rho(dist,dc) # computes density for each point according to distance matrix
    delta = compute_delta(dist,rho) # computes delta for each point according to density and distance matrix
    nclus,cluslabels,centid,threshold = find_centroids_and_cluster(dist,rho,delta,alpha) # automatically detect cluster centroids and clusters
    halolabels = halo_assign(dist,cluslabels,centid,ishalo) # removes peripheral points of the cluster
    return rho,delta,centid,cluslabels,halolabels,threshold
    
    
def compute_rho(dist,dc):
    """
    Parameters
    ----------
    dist : is a nbins x nbins matrix of distances between data bins
    dc : average percentage of neighbours, ranging from [0, 1]

    Returns
    -------
    rho : nbins x 1 vector. local density of point i

    """
    npts = len(dist)
    prctneig = int(np.round(dc*npts)) # number of closest neighbors to compute density
    dist_sorted = np.sort(dist,axis=0) # sorting each row in asciending order
    rho =  1/np.mean(dist_sorted[1:prctneig+1,:],axis=0) # density computation
    rho[np.isnan(rho)]=0 # setting nans to 0 density
    return rho
    
def compute_delta(dist, rho):
    """
    % DENSITYCLUST Clustering by fast search and find of density peaks.
    %   SEE the following paper published in *SCIENCE* for more details:
    %       Alex Rodriguez & Alessandro Laio: Clustering by fast search and find of density peaks,
    %       Science 344, 1492 (2014); DOI: 10.1126/science.1242072.
    %   INPUT:
    %       dist: is a nbins x nbins matrix of distances between data bins
    %       rho:  nbins x 1 vector. local density of point i
    %   OUTPUT:
            delta: minimun distance to the next higher density point
    
    %
    % RUBEN HERZOG 2023.
    % 
    """
    rho_sort_id = np.argsort(rho) # index to sort
    rho_sort_id = (rho_sort_id[::-1]) # reversing sorting indexes
    sort_rho = rho[rho_sort_id] # sortig rho in ascending order
    gtmat = np.greater_equal(sort_rho,sort_rho[:,None]) # gtmat(i,j)=1 if rho(i)>=rho(j) and 0 otherwise
    
    sortdist = np.zeros_like(dist) 
    sortdist = dist[rho_sort_id,:] # sorting distances according to rho
    sortdist = sortdist[:,rho_sort_id]
    
    seldist = gtmat*sortdist # keeping only distance to points with highest or equal rho 
    seldist[seldist==0] = float("inf") 
              
    auxdelta = np.min(seldist,axis=1)
    delta=np.zeros_like(auxdelta) 
    delta[rho_sort_id] = auxdelta 
    delta[rho==np.max(rho)] = np.max(delta[np.logical_not(np.isinf(delta))]) # assigns max delta to the max rho
    delta[np.isinf(delta)] = 0
    return delta
    
def find_centroids_and_cluster(dist,rho,delta,alpha):
    """
    Finds centroids based on the rho vs delta relationship. 
    Fits a power law to the rho vs delta function and uses the alpha-th
    prediction bound as threshold for choosing the centroids.
    Then, the non-centroids are assigned to their closest centroid.
    Parameters
    ----------
    dist : is a nbins x nbins matrix of distances between data bins
    rho : nbins x 1 vector. local density of point i
    delta : minimun distance to the next higher density point
    alpha : Scalar. percentile to compute prediction bounds on the centroid detection

    Returns
    -------
    nclus : Scalar. Number of clusters
    cluslabels : Vector with label id for each point.
    centidx : centroid id.
    threshold : threshold curve used to detect centroids

    """
    npnts = len(rho)    
    centid = np.zeros((npnts))    

    # fitting a power law to the rho vs delta relationship
    # preparing data
    mindelta = 10**(-6) # min delta to be considered, improves fit
    nzind = np.where(np.logical_and(delta>mindelta, rho>0))[0] # uses only delta different from 0 and rhos higher than 0
    nzind = np.intersect1d(nzind, np.where(np.logical_not(np.isinf(rho))))
    nzdelta = delta[nzind] # y of fit
    nzrho = rho[nzind] # x of fit
    
    # fitting a line in log space
    threshold = np.exp(estimate_threshold(np.log(nzrho),np.log(nzdelta),alpha)) # to linear form
    # threshold = np.maximum(np.exp(threshold),np.ones_like(threshold)*np.max(delta)*0.3) # to linear form
    
    # selecting centroids
    selid = (nzdelta>threshold) # delta larger than threshold    
    auxid = nzind[selid] # centroids on original basis
    if len(auxid)==0: # if no centroids are detected
        auxid=(np.argmax(rho)) # centroids are the points with the larger rho
        nclus = 1
    else:
        nclus = len(auxid)
    centid[auxid] = np.arange(0,nclus,1)+1 # assigning labels to centroids. 0 is used later for non-assigned points
    threshold = np.vstack((nzrho,threshold)) # saving the x and y of the threshold
    
    # assigning points to clusters based on their distance to the centroids
    if nclus==1: # if only one cluster is found
        cluslabels = np.ones(npnts)
        centidx = np.where(centid)[0] # centroid are the points with larger rho
    else:
        centidx = np.where(centid)[0] # index of centroids
        dist2cent = dist[centidx,:] # distance of each points to the centroids
        cluslabels = np.argmin(dist2cent,axis=0)+1
        _,cluscounts = np.unique(cluslabels,return_counts=True) # number of elements of each cluster
        one_mem_clus = np.where((cluscounts==1) | (cluscounts==0))[0] # index of clusters with 0 or 1 members
        if one_mem_clus.size>0: # if there is one or more cluster with 0 or 1 member
            clusidx=np.delete(centidx,one_mem_clus) # removing
            centid = np.zeros(len(centid))
            nclus = nclus-len(one_mem_clus) # subsatracts these clusters
            centid[clusidx]=np.arange(0,nclus,1)+1 # re labeling centroids            
            dist2cent = dist[centidx,:]# re compute distances from centroid to any other point
            cluslabels = np.argmin(dist2cent,axis=0)+1 # re assigns clusters 
            
    return nclus,cluslabels,centidx,threshold
    

def halo_assign(dist,cluslabels,centidx,op):
    """
    Removes peripheral points from clusters based on a threshold defined by the average distance to the cluster centroid
    Parameters
    ----------
    dist : is a nbins x nbins matrix of distances between data bins
    cluslabels : Vector with label id for each point.
    centidx : centroid id.
    op : Logical. If op==1, removes from each cluster all the points that are farther than
    the average distance to the cluster centroid

    Returns
    -------
    halolabels : cluster labels after removal of peripheral points

    """
    halolabels = cluslabels.copy()
    if op:
        # sameclusmat[i,j]=1 is i and j belongs to the same cluster and 0 otherwise
        sameclusmat = np.equal(cluslabels,cluslabels[:,None]) #
        sameclus_cent = sameclusmat[centidx,:] # selects only centroids
        dist2cent = dist[centidx,:] # distance to centroids
        dist2cluscent = dist2cent*sameclus_cent # preserves only distances to the corresponding cluster centroid
        nclusmem = np.sum(sameclus_cent,axis=1) # number of cluster members
        
        meandist2cent = np.sum(dist2cluscent,axis=1)/nclusmem # mean distance to corresponding centroid
        gt_meandist2cent = np.greater(dist2cluscent,meandist2cent[:,None]) # greater than the mean dist to centroid
        remids = np.sum(gt_meandist2cent,axis=0)
        halolabels[remids>0] = 0 # setting to 0 the removes points
        return halolabels


def estimate_threshold(x,y,alpha):
    """
    Fits a regression between x and y and computes the upper confidence interval based on 'alpha'

    Parameters
    ----------
    x : vector
    y : vector, same size than y
    alpha : alpha for confidence interval. Should be close to 1

    Returns
    -------
    threshold : upper bound of the regression confidence interval based on alpha

    """
    x = sm.add_constant(x)
    model = sm.OLS(y,x)
    results = model.fit()
    _,_,threshold = wls_prediction_std(results,alpha=alpha)
    return threshold

    
def linear_regression(x, y, prob):
    """
    Return the linear regression parameters and their <prob> confidence intervals.
    ex:
    >>> linear_regression([.1,.2,.3],[10,11,11.5],0.95)
    b0 is intercetp, b1 is slope, bb0 is interval for b0 and bb1 for b1
    NOT REALLY GOOD AT ESTIMATING BOUNDS
    """
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    xy = x * y
    xx = x * x

    # estimates
    b1 = (xy.mean() - x.mean() * y.mean()) / (xx.mean() - x.mean()**2)
    b0 = y.mean() - b1 * x.mean()
    s2 = 1./n * sum([(y[i] - b0 - b1 * x[i])**2 for i in range(n)])
    #print 'b0 = ',b0
    #print 'b1 = ',b1
    #print 's2 = ',s2
    
    #confidence intervals    
    alpha = 1 - prob   
    c = -1 * scipy.stats.t.ppf(alpha/2.,n-2)
    bb1 = c * (s2 / ((n-2) * (xx.mean() - (x.mean())**2)))**.5
    #print 'the confidence interval of b1 is: ',[b1-bb1,b1+bb1]
    
    bb0 = c * ((s2 / (n-2)) * (1 + (x.mean())**2 / (xx.mean() - (x.mean())**2)))**.5
    #print 'the confidence interval of b0 is: ',[b0-bb0,b0+bb0]
    return b1,b0,bb1,bb0




def plot_clustering_summary(delta,rho,centid,cluslabels,threshold,pcs,color):
    
    offset = 0 # in case of no 0 cluster
    fig=plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122,projection='3d')
    ax1.plot(rho,delta,'.')
    if np.min(cluslabels)!=0:
        offset = 1
        
    for clus,c in enumerate(color):
        ax1.plot(rho[centid[clus]],delta[centid[clus]],'o',c=c)
        ax2.plot(pcs[cluslabels==(clus+offset),0],
                 pcs[cluslabels==(clus+offset),1],
                 pcs[cluslabels==(clus+offset),2],'.',c=c)

    ax1.plot(threshold[0,:],threshold[1,:],'r.')
    ax1.set_xlabel('Density (\rho)')
    ax1.set_ylabel('Distance (\delta)')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')


def clustering_subsample(rand_ids,data,numpcs,dc,ci_alpha,ishalo):

    N,T = np.shape(data)
    data = data.astype('f') # to float32 to reduce memory usage
    subdata = data[:,rand_ids] # subsampling
    # PCA and distance matrix
    distmat,proj_data = pca_and_distance(subdata,numpcs)
    # pcac = PCA(n_components=numpcs)
    # proj_data = pcac.fit_transform(subdata.T)
    # Euclidean distances matrix on PC space
    # distmat = distance.cdist(proj_data, proj_data, 'euclidean')
    # Density based clustering
    rho,delta,centid,_,halolabels,_ = clustering_by_density(distmat,dc,ci_alpha,ishalo)
    nens = len(centid)
    tmpcent = np.zeros((N,nens))
    for n in range(nens):
        tmpcent[:,n] = np.nanmean(subdata[:,halolabels==(n+1)],axis=1) # proto-template for each cluster
    
    # samp_cent.append(tmpcent)
    print('Found '+str(nens),' clusters')
    return tmpcent



def fit_templates_to_data(data,templates,ccthr):
    """
    Fit templates to data and assigns each data point to the template with maximal correlation.
    If correlation is belos ccthr, does not assigns templates

    Parameters
    ----------
    data : N x T data matrix to be fit
    templates : N x ntemps with templates to fit
    ccthr : correlations threshold. If corr(data,template)<ccthr, no template is assigned

    Returns
    -------
    temp_seq : temporal sequence of each template
    max_corr : maximal correlation at each time point
    temp_corr : correlation of each template to data
    """
    _,ntemps = np.shape(templates)
    temp_corr = 1-distance.cdist(data.T, templates.T, 'correlation')
    temp_corr[np.isnan(temp_corr)] = 0
    temp_seq = np.argmax(temp_corr,axis=1)+1
    max_corr = np.max(temp_corr,axis=1)
    temp_seq[max_corr<ccthr] = 0
    return temp_seq,max_corr,temp_corr
    
def cluster_epoch_subsamples(rand_subs_id,files,nrand_epochs,epoch_size,nchan,dc,numpcs,ci_alpha,ishalo):
      
    nsubs = len(files)
    if len(rand_subs_id)==1: # uses all subjects
       rand_subs_id = range(nsubs) 
    start = time.time()
    warnings.filterwarnings("ignore")
    # Initial parameters
    nrand_subs = len(rand_subs_id)
    sel_size = nrand_epochs*nrand_subs
    nnplets = int((nchan*(nchan-1)/2 + 1)) # pairs + unique triplet
    tc = np.zeros((nnplets,sel_size))
    dtc = np.zeros((nnplets,sel_size))
    o_info= np.zeros((nnplets,sel_size))
    s_info = np.zeros((nnplets,sel_size))
    
    # Random epochs from random subjects  
    sel_epochs = np.zeros((sel_size,nchan,epoch_size))    
    for (s,subid) in enumerate(rand_subs_id):
        
        epochs = mne.read_epochs(files[subid], preload = True,verbose='ERROR')
        epochs = epochs._data[:,:3,:]
        this_nepochs,_,_ = np.shape(epochs)
        rand_epochs_id = np.array(random.sample(range(this_nepochs), nrand_epochs))
        sel_epochs[(s*nrand_epochs):((s*nrand_epochs)+nrand_epochs),:,:] = epochs[rand_epochs_id,:,:]
       
    # Computing MVIT measures
    # HERE REPLACE WITH ANY FEATURE TO CHARACTERIZE DATA
    # e.g. spectral power, hctsa
    for e in range(sel_size):
        tc[:,e],dtc[:,e],o_info[:,e],s_info[:,e],_,_= mvit.multi_order_meas(sel_epochs[e,:,:].T)

    mvit_meas = np.vstack((tc,dtc[3],o_info[3],s_info[3]))
    mvit_meas[np.isnan(mvit_meas)] = 0
    mvit_meas[np.isinf(mvit_meas)] = 0
    nmeas,_ = np.shape(mvit_meas)
    
    # PCA and distance matrix
    distmat,proj_data = pca_and_distance(mvit_meas,numpcs)
    # pcac = PCA(n_components=numpcs)
    # proj_data = pcac.fit_transform(mvit_meas.T)
    # Euclidean distances matrix on PC space
    # distmat = distance.cdist(proj_data, proj_data, 'euclidean')
    # Clustering
    _,_,_,_,halolabels,_ = clustering_by_density(distmat,dc,ci_alpha,ishalo)
    # Extracting templates
    nclust = np.max(halolabels)
    tmpcent = np.zeros((nmeas,nclust))
    for n in range(nclust):
        tmpcent[:,n] = np.nanmean(mvit_meas[:,halolabels==(n+1)],axis=1) # proto-template for each cluster
    
    print(str(nclust)+' clusters found in '+"{:.2f}".format(time.time() - start)+' s')
    
    return tmpcent


def cluster_epoch_features(rand_subs_id,files,nfeats,nrand_epochs,epoch_size,nchan,dc,numpcs,ci_alpha,ishalo):
      
    nsubs = len(files)
    if rand_subs_id==0: # uses all subjects
       rand_subs_id = range(nsubs) 
    start = time.time()
    warnings.filterwarnings("ignore")
    # Initial parameters
    rand_subs_id = range(nsubs)
    nrand_subs = len(rand_subs_id)
    sel_size = nrand_epochs*nrand_subs   
    # Random epochs from random subjects  
    sel_epochs = np.zeros((nfeats,sel_size))    
    for (s,subid) in enumerate(rand_subs_id):
        
        epochs = loadmat(files[subid])
        epochs = epochs["epoch_feats"]
        _,this_nepochs = np.shape(epochs)
        rand_epochs_id = np.array(random.sample(range(this_nepochs), nrand_epochs))
        sel_epochs[:,(s*nrand_epochs):((s*nrand_epochs)+nrand_epochs)] = epochs[:,rand_epochs_id]
   
    # Normalization, PCA and distance matrix
    distmat,proj_data = pca_and_distance(sel_epochs,numpcs)
    # Clustering
    _,_,_,_,halolabels,_ = clustering_by_density(distmat,dc,ci_alpha,ishalo)
    # Extracting templates
    nclust = np.max(halolabels)
    tmpcent = np.zeros((nfeats,nclust))
    for n in range(nclust):
        tmpcent[:,n] = np.nanmean(sel_epochs[:,halolabels==(n+1)],axis=1) # proto-template for each cluster
    
    # Removing nan or full of zeros
    sum_op = np.sum(tmpcent,axis=0)
    not_zero = sum_op>0
    not_nan = np.logical_not(np.isnan(sum_op))
    valid_tmp = np.logical_and(not_zero,not_nan)
    tmpcent = tmpcent[:,valid_tmp]
    
    print(str(np.sum(valid_tmp))+' clusters found in '+"{:.2f}".format(time.time() - start)+' s')
    
    return tmpcent


def pca_and_distance(data,numpcs,op=True):
    """
    

    Parameters
    ----------
    data : Nfeatures x nsamples
    numpcs : number of principal components to use

    Returns
    -------
    dtsmat :  nsamples x nsamples distance matrix.
    proj_data : data projected on the numpcs principal components

    """
    # Removing nans and infs
    data[np.isnan(data)]=0
    data[np.isinf(data)]=0
    # normalization of data
    norm_data = data
    if op:
        norm_data = zscore(data,axis=1)

    pcac = PCA(n_components=numpcs)
    proj_data = pcac.fit_transform(norm_data.T)
    # Euclidean distances matrix on PC space
    distmat = distance.cdist(proj_data, proj_data, 'euclidean')
    return distmat, proj_data

if __name__=='__main__':
    
    find_centroids_and_cluster(dist,rho,delta,alpha)

