import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import spearmanr
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import os
from os import system
import sys
sys.path.append('../')


from matplotlib.lines import Line2D
from lib.ml_wrapper import asynch_reconstruction
import lib.figs_funcs as figfunc
import lib.funcs_raster as funcs_raster
import lib.funcs_general as funcs_general
import lib.fun_plotting as fun_plotting
import lib.funcs_ko as funcs_ko


# ------------------------------ Grid Search for the best set of Hyperparameters

def grid_search(spins, params, interaction_list, genes_order, Ntrials=5, seedSet=20961, Norm=True, thr=0.0, frac_thr = False):
    """Evaluate the performance of the model with different sets of hyperparameters

    Args:
       - spins (numpy array): matrix of the binnarized data, rows are genes, columns are time points
       - params (dictionary): possible set of hyperparameters to evaluate
       - interaction_list (list pf strings): list of known interactions with the format [gene1, gene2, interaction_type]
       - genes_order (numpy array): list of genes in the same order as the rows of the DataFrame
       - Ntrials (int, optional): Number of set of hyperparameters selected. Defaults to 5.
       - seedSet (int, optional): seed
       - Norm (bool, optional): Normalize the interactiona. Defaults to True.
       - thr (float, optional): threshold to set an interactions to zero if below it. Defaults to 0.0.
        
    output: 
        matx_sel: matrix of the inferred interaction matrices
        tp_val: array of the true positive values for each set of hyperparameters
        info_int: array of the information about the interactions for each set of hyperparameters with the format:
                   - row 0: who acts as a regulator
                   - row 1: who acts as a target
                   - row 2: inferred interaction value
                   - row 3: true interaction if set to 1, 0 otherwise.
    """
    np.random.seed(seedSet)   
    
    matx_sel = np.zeros((Ntrials, len(genes_order), len(genes_order)))
    info_int  = np.zeros((4, len(interaction_list), Ntrials))
    tp_val  = np.zeros(Ntrials)
    for ii in range(Ntrials):
        print("\n\nModel #", ii+1) 
        par_sel = {}
        for jj in params.keys():
            par_sel[jj] = np.random.choice(params[jj])
        print("Params", par_sel)    

        # initialize the reconstruction 
        model = asynch_reconstruction(spins, delta_t = 1, LAMBDA = par_sel["LAMBDA"], MOM = par_sel["MOM"], 
                                      opt = par_sel["opt"], reg = par_sel["reg"],
                                     ax_names = genes_order) 

        # reconstruct the model 
        model.reconstruct(spins, Nepochs = par_sel["Nepochs"], start_lr = par_sel["lr"], 
                          drop = par_sel["drop"], edrop = par_sel["edrop"])
        
        ###############################
        if frac_thr:
            thr = np.percentile(np.abs(model.J), thr*100)/np.nanmax(np.abs(model.J))
            print("computed threshold", thr)
        ###############################

        # evaluate the model with whant we know experimentally
        tp_val[ii], info_int[:,:,ii], _ = funcs_general.TP_plot(interaction_list, model.J, genes_order, 
                                                   inferred_int_thr=thr, Norm_Matx = False,
                                                   data_type="scRNA-seq PST MB with ISING",
                                                   figplot=False, verbose=True, nbin=30, Norm=Norm)

        #save the inferred interaction matrix
        matx_sel[ii, :, :] = model.J

    return(matx_sel, tp_val, info_int)


#### Computing simulated data and its Pearson correlation
def grid_search_noPrior(spins, params, interaction_list, genes_order, Ntrials=5, seedSet=20961, Norm=True, thr=0.0, N_sim=10, N_ts=9000, fig_sim = False, spins_shuffled=[]):
    """
    """
    np.random.seed(seedSet)   
    
    matx_sel = np.zeros((Ntrials, len(genes_order), len(genes_order)))
    info_int  = np.zeros((4, len(interaction_list), Ntrials))
    tp_val  = np.zeros(Ntrials)
    dist  = np.zeros(Ntrials)
    
    for ii in range(Ntrials):
        print("\n\nModel #", ii+1) 
        par_sel = {}
        np.random.seed(ii+seedSet)
        for jj in params.keys():
            par_sel[jj] = np.random.choice(params[jj])
        print("Params", par_sel)  
  

        # initialize the reconstruction 
        model = asynch_reconstruction(spins, delta_t = 1, LAMBDA = par_sel["LAMBDA"], MOM = par_sel["MOM"], 
                                      opt = par_sel["opt"], reg = par_sel["reg"],
                                     ax_names = genes_order) 

        # reconstruct the model 
        model.reconstruct(spins, Nepochs = par_sel["Nepochs"], start_lr = par_sel["lr"], 
                          drop = par_sel["drop"], edrop = par_sel["edrop"])

        # evaluate the model with whant we know experimentally
        tp_val[ii], info_int[:,:,ii], _ = funcs_general.TP_plot(interaction_list, model.J, genes_order, 
                                                   inferred_int_thr=thr, Norm_Matx = False,
                                                   data_type="scRNA-seq PST MB with ISING",
                                                   figplot=False, verbose=False, nbin=30, Norm=Norm)

        # evaluate the model with the comparison between original and simulated Pearson correlation matrix
        spins_new_lN = np.zeros((spins.shape[0], N_ts, N_sim))
        # generate N_sim new time series of spins using model
        for ll in range(N_sim):
            np.random.seed(ll+1234)
            spins_new_lN[:,:,ll] = model.generate_samples(seed=ll*2, t_size=N_ts)
        # compute the Pearson correlation matrix
        cm_sim_lN = np.zeros((spins_new_lN.shape[0], spins_new_lN.shape[0], spins_new_lN.shape[2]))
        for kk in range(spins_new_lN.shape[2]):
            cm_sim_lN[:,:, kk] = np.corrcoef(spins_new_lN[:,:,kk])
        cm_sim_lN_mean = np.nanmean(cm_sim_lN, axis=2) # mean for simulated correlation matrix
            
        cm_original_lN = np.corrcoef(spins) # original correlation matrix
        
        if fig_sim:
            # fun_plotting.raster_plot(spins_new_lN[:,:,0], 'Reconstruction', 1, genes_order)
            # plt.show()
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
            # figfunc.plotmat(corr_matxs_rnd_lN_noDiag, fig, ax[0], genes_order, "Random data")
            figfunc.plotmat(cm_original_lN, fig, ax[0], genes_order, "Original data", fix = True)
            figfunc.plotmat(cm_sim_lN_mean, fig, ax[1], genes_order, "ISING Simulated data", fix = True)
            plt.show()
            
        # random correlation matrix: shuffle the rows and columns of the original matrix Ntrials times
        corr_matxs_rnd_lN = np.array([np.corrcoef(spins_shuffled[i,:,:]) for i in range(spins_shuffled.shape[0])])
        noise_dist = np.mean([funcs_ko.sum_squared_abs_diff(cm_original_lN, corr_matxs_rnd_lN[i,:,:]) for i in range(50)])
        # compute the  weighted distance between the original and simulated matrixes
        dist[ii] = np.mean([funcs_ko.sum_squared_abs_diff(cm_original_lN, cm_sim_lN[:,:,i]) for i in range(cm_sim_lN.shape[2])])/ noise_dist # normalize by the noise
        if  np.isnan(cm_sim_lN).sum() > (24*24*spins_new_lN.shape[2])-24*24*0.5*spins_new_lN.shape[2]:
            dist[ii] =np.nan
        #save the inferred interaction matrix
        matx_sel[ii, :, :] = model.J
        print("TP value", tp_val[ii], "dist", dist[ii])
        

    return(matx_sel, tp_val, info_int, dist)

