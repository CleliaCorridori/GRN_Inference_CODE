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

