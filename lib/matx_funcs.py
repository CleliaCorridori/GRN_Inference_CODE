import pandas as pd
import numpy as np
import scipy as sp
import math
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sn

def randomize_df(df, seed):
    df_rnd = df.unstack().rename_axis(['columns', 'index']).reset_index(name = 'value')
    df_rnd = df_rnd.sample(frac=1, random_state = seed).reset_index(drop=True) #shuffle
    df_rnd = pd.DataFrame(np.reshape(np.array(df_rnd["value"]), (df.shape)))
    return(df_rnd)

def z_score(vect1, vect2):
    """ Z-score between two arrays"""
    mean1 = np.mean(np.ndarray.flatten(vect1))
    mean2 = np.mean(np.ndarray.flatten(vect2))
#     print("mean: ", mean1, mean2)
    
    std1 = np.var(np.ndarray.flatten(vect1))
    std2 = np.var(np.ndarray.flatten(vect2))
#     print("std: ", std1, std2)
        
    z_sc = (mean1 - mean2)/np.sqrt(std1+std2)
    return(z_sc)

def BBmatrix(cm):
    """Function to compute the interaction matrix using the Barzel and Barabasi method
    INPUT: cm is the correlation matrix computed with Pearson"""
    BBmatx = np.matmul(cm - np.identity(np.shape(cm)[0]) + np.diag(np.diag(np.matmul(cm - np.identity(np.shape(cm)[0]),cm))),np.linalg.pinv(cm))
    return(BBmatx)

def MEmatrix(cm):
    """Function to compute the interaction matrix using the Maximum Entropy method
    INPUT: cm is the correlation matrix computed with Pearson
    The minus is here because of eq 18 of paper "Inferring Pairwise Interactions from Biological Data Using Maximum-Entropy     Probability Models", Stein """
    MEmatrix = -np.linalg.pinv(cm)
    return(MEmatrix)

def corr_matxs_comp(df, N):
    """Function to shuffle the dataframe N times and compute each time the correlation matrix (Pearson)
    output dim: corr-dim1 x corr-dim2 x N"""
    cms_rnd = np.zeros((df.shape[0]*df.shape[0], N))
    for ii in range(N):
        df_rnd = randomize_df(df, ii)
        cms_rnd[:,ii]= np.ndarray.flatten(np.corrcoef(df_rnd))
    return(cms_rnd)

def corr_matxs_compMF(df, N):
    """Function to shuffle the dataframe N times and compute each time the correlation matrix (Pearson)
    output dim: corr-dim1 x corr-dim2 x N
    !!!preserving the 24x24xN shape"""
    cms_rnd = np.zeros((df.shape[0],df.shape[0], N))
    for ii in range(N):
        df_rnd = randomize_df(df, ii+1000)
        cms_rnd[:,:,ii]= np.corrcoef(df_rnd)
        np.fill_diagonal(cms_rnd[:,:,ii], 0)
    return(cms_rnd)

def corr_matxs_comp_Nan(df, N):
    """Function to shuffle the dataframe N times and compute each time the correlation matrix (Pearson)
    output dim: corr-dim1 x corr-dim2 x N
    Nan in diagonal elements"""
    cm_rnd = np.zeros((df.shape[0],df.shape[0]))
    cms_rnd = np.zeros((df.shape[0]*df.shape[0], N))
    for ii in range(N):
        df_rnd = randomize_df(df, ii)
        cm_rnd = np.corrcoef(df_rnd)
        np.fill_diagonal(cm_rnd, float("Nan"))
        cms_rnd[:,ii]= np.ndarray.flatten(cm_rnd)
    return(cms_rnd)

def BB_matxs_comp(df, N):
    """Function to shuffle the dataframe N times and compute each time the Barzel-Barabasi matrix
    output dim: matx-dim1 x matx-dim2 x N"""
    cms_rnd = np.zeros((df.shape[0]*df.shape[0], N))
    for ii in range(N):
        df_rnd = randomize_df(df, ii)
        cms_rnd[:,ii]= np.ndarray.flatten(BBmatrix(np.corrcoef(df_rnd)))
    return(cms_rnd)

def ME_matxs_comp(df, N):
    """Function to shuffle the dataframe N times and compute each time the MaxEnt matrix
    output dim: matx-dim1 x matx-dim2 x N"""
    cms_rnd = np.zeros((df.shape[0]*df.shape[0], N))
    for ii in range(N):
        df_rnd = randomize_df(df, ii)
        cms_rnd[:,ii]= np.ndarray.flatten(MEmatrix(np.corrcoef(df_rnd)))
    return(cms_rnd)

def z_scores_comp(matx, N):
    "Function to compute the Z-scores between vectors couples. The Z-scores array have dim=#combinations(#vect,2)"
    z_list_rnd = np.zeros(math.comb(N, 2))
    ll=0
    for jj in range(matx.shape[1]):
        for kk in range(jj+1, matx.shape[1]):
            z_list_rnd[ll] = z_score(matx[:,jj], matx[:,kk])
            ll+=1
    return(z_list_rnd)

def plot_zscore(z, Nsigma=2):
    mu, sigma = sp.stats.norm.fit(z)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    plt.figure(figsize=(8, 4))
    sn.histplot(z, stat="density", bins="fd") #, label = "data")
    plt.plot(x, stats.norm.pdf(x, mu, sigma), color="Blue") #, label = "fit")
    plt.xlabel("Z-score")
    plt.vlines(mu-Nsigma*sigma, 0, np.max(stats.norm.pdf(x, mu, sigma)+10), color="darkblue", label = str(Nsigma)+"$\sigma$")
    plt.vlines(mu+Nsigma*sigma, 0, np.max(stats.norm.pdf(x, mu, sigma)+10), color="darkblue") #, label = "2$\sigma$")
 