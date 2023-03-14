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

from statsmodels.stats.weightstats import DescrStatsW


from matplotlib.lines import Line2D
from lib.ml_wrapper import asynch_reconstruction
import lib.figs_funcs as figfunc
import lib.funcs_raster as funcs_raster
import lib.fun_plotting as fun_plotting

# ------------------------------
# ------ GENERAL FUNCTIONS ------
path = "/Users/cleliacorridori/Dropbox_2021 Dropbox/Jorah Mormont/GRN_Inference/DATA/" # for Mac

# genes of OUR dataset
genes_order = np.loadtxt(path+"general_info/genes_order.csv", dtype="str") 
# time steps for the dataset
time=["00h", "06h", "12h", "24h", "48h"]
# Genes Classification
naive = ["Klf4", "Klf2", "Esrrb", "Tfcp2l1", "Tbx3", "Stat3", "Nanog", "Sox2"]
formative = ["Nr0b1", "Zic3", "Rbpj", "Utf1", "Etv4", "Tcf15"]
committed = ["Dnmt3a", "Dnmt3b", "Lef1", "Otx2", "Pou3f1", "Etv5"]


# -------------------------------------------------------------------------------------------------
# ------------------------------ Plot average activity time of genes ------------------------------
# -------------------------------------------------------------------------------------------------

def plot_activity_simulated(spins_df_sim, genes_order, title, color, ax):
    # spins_df_sim is (n_genes, n_time, n_test)
    avg_activity_each     = spins_df_sim.mean(axis=1)
    avg_activity_std_each = spins_df_sim.std(axis=1, ddof=1)/np.sqrt(spins_df_sim.shape[1])
    # print(avg_activity_each.shape, avg_activity_std_each.shape)
    avg_activity = np.zeros(spins_df_sim.shape[0])
    avg_activity_std = np.zeros(spins_df_sim.shape[0])
    for j in range(spins_df_sim.shape[0]):
        # avg_activity[j] = DescrStatsW(avg_activity_each[j,:], weights=1/(avg_activity_std_each[j,:])**2, ddof=1).mean
        # avg_activity_std[j] = DescrStatsW(avg_activity_each[j,:], weights=1/(avg_activity_std_each[j,:])**2, ddof=1).std
        avg_activity[j] = np.mean(avg_activity_each[j,:])
        avg_activity_std[j] = np.std(avg_activity_each[j,:], ddof=1)
    
    ax.errorbar(genes_order, avg_activity, yerr=avg_activity_std, 
                 alpha=1, 
                 fmt="o", ms = 10,
                 elinewidth=1,
                 color=color,
                 capsize=10,
                 label = title)
    ax.legend(loc="upper left", fontsize=16)
    # ax.set_xticks(fontsize=12)
    ax.set_ylabel("Average spin", fontsize=16)
    ax.set_xlabel("Genes", fontsize=16)
    ax.set_title(title, fontsize=20)
    ax.grid(True)
    return(avg_activity, avg_activity_std)

def plot_activity(spins_df, genes_order, title, color, ax):
    # spins_df is (n_genes, n_time)
    avg_activity     = spins_df.mean(axis=1)
    avg_activity_std = spins_df.std(axis=1)/np.sqrt(spins_df.shape[1])
    ax.errorbar(genes_order, avg_activity, yerr=avg_activity_std, 
                 alpha=1, 
                 fmt="o", ms = 10,
                 elinewidth=1,
                 color=color,
                 capsize=10,
                 label = title)
    ax.legend(loc="upper left", fontsize=16)
    # ax.set_xticks(fontsize=12)
    ax.set_ylabel("Average spin", fontsize=16)
    ax.set_xlabel("Genes", fontsize=16)
    ax.set_title(title, fontsize=20)
    ax.grid(True)
    return(avg_activity, avg_activity_std)

# -------------------------------------------------------------------------------------------------
# ------------------------------------- Distance between matrices ---------------------------------
# -------------------------------------------------------------------------------------------------


def sum_squared_abs_diff(array1, array2):
    """Calculate the sum of squared absolute differences between two matrices"""
    diff = (array1.flatten()-array2.flatten())**2
    return np.sqrt(np.sum(diff))

# -------------------------------------------------------------------------------------------------
# ------------------------------------- KO: knockout  functions -----------------------------------
# -------------------------------------------------------------------------------------------------

def info_KO(matx,model, KO_gene="Rbpj", genes_order=genes_order, multiple=False):
    """Remove the KO_gene from the interaction matrix and from the field"""
    if multiple:
        KO_gene_idk = [np.where(genes_order == KO_gene[i])[0][0]  for i in range(len(KO_gene))]
    else:
        KO_gene_idk = np.where(genes_order == KO_gene)[0] 
    KO_rec_matx = np.delete(matx, KO_gene_idk, axis=0)
    KO_rec_matx = np.delete(KO_rec_matx, KO_gene_idk, axis=1)
    KO_rec_field = np.delete(model.h, KO_gene_idk, axis=0)
    KO_genes_order = np.delete(genes_order, KO_gene_idk, axis=0)
    return(KO_rec_matx, KO_rec_field, KO_gene_idk, KO_genes_order)

def KO_plots_oneSim(ko_spins, ko_avg, ko_std, wt_avg, wt_std, ko_genes_order, raster=True, avg=True):
    """function to simulate the KO data (active/inactive genes in time) and to plot, depending on the decision of the user, 
    the raster plot and the average active time for each gene in wild type and in KO.
    Args:
        ko_spins (array): array of the KO data (active/inactive genes in time)
        ko_avg (array): array of the average active time for each gene in KO
        ko_std (array): array of the standard deviation of the active time for each gene in KO
        wt_avg (array): array of the average active time for each gene in wild type
        wt_std (array): array of the standard deviation of the active time for each gene in wild type
        ko_genes_order (array): array of the genes in the same order of the KO data
        
        raster (bool): True if the user wants to plot the raster plot
        avg (bool): True if the user wants to plot the average active time for each gene in wild type and in KO
    """
    # raster plot
    if raster:
        fun_plotting.raster_plot(ko_spins, 'Reconstruction', 1, ko_genes_order)
        plt.show()
        
    # average activity time per gene in wild type and in KO
    if avg:
        plt.figure(figsize=(18,5))
        plt.errorbar(ko_genes_order, ko_avg, yerr=ko_std,  
                     alpha=1, 
                     fmt="o", ms = 10,
                     elinewidth=3,
                     color="steelblue",
                     capsize=10,
                     label= "Simulated KO Data")

        plt.errorbar(ko_genes_order, wt_avg, yerr=wt_std,
                     alpha=1, 
                     fmt="o", ms = 10,
                     elinewidth=1,
                     color="indianred",
                     capsize=10,
                     label = "Simulated WT Data")
        plt.legend(loc="upper left", fontsize=16)
        plt.xticks(fontsize=12)
        plt.ylabel("Average spin", fontsize=16)
        plt.xlabel("Genes", fontsize=16)
        plt.title("Average spin values for each genes", fontsize=20)
        plt.grid(True)
        plt.show()
    
def KO_avg_weighted(matx, field, wt_spins, model, N_test_KO=100):
    """ Compute the average spins value for each gene in KO and in WT
    Args:
        matx (numpy array): interaction matrix
        field (numpy array): field values
        genes_order (list of strings): list of genes in the order of the interaction matrix
        
    """
    # average activity for each gene in WT using original data
    # wt_avg = np.array(wt.mean(axis=1))
    # wt_std = np.array(wt.std(axis=1, ddof=1))/np.sqrt(wt.shape[1])

    # activity for each gene in KO
    KO_avg_spin = np.zeros((matx.shape[0], N_test_KO))
    KO_std_spin = np.zeros((matx.shape[0], N_test_KO))
    for i in range(N_test_KO):
        KO_spins = model.generate_samples_SetData(matx=matx, field=field, seed=i*5)+1
        KO_avg_spin[:,i] = np.array(KO_spins.mean(axis=1))
        KO_std_spin[:,i] = np.array(KO_spins.std(axis=1, ddof=1))/np.sqrt(KO_spins.shape[1])
    
    KO_weighted_avg = np.zeros(KO_avg_spin.shape[0])
    KO_weighted_std = np.zeros(KO_avg_spin.shape[0])    
    for j in range(KO_avg_spin.shape[0]):
        KO_weighted_avg[j] = DescrStatsW(KO_avg_spin[j,:], weights=1/(KO_std_spin[j,:])**2, ddof=1).mean
        KO_weighted_std[j] = DescrStatsW(KO_avg_spin[j,:], weights=1/(KO_std_spin[j,:])**2, ddof=1).std
        
    # activity for each gene in WT
    wt_avg_spin = np.array(wt_spins.mean(axis=1))
    wt_std_spin = np.array(wt_spins.std(axis=1, ddof=1))/np.sqrt(wt_spins.shape[1])
    wt_weighted_avg = np.zeros(wt_avg_spin.shape[0])
    wt_weighted_std = np.zeros(wt_avg_spin.shape[0])    
    for j in range(wt_avg_spin.shape[0]):
        wt_weighted_avg[j] = DescrStatsW(wt_avg_spin[j,:], weights=1/(wt_std_spin[j,:])**2, ddof=1).mean
        wt_weighted_std[j] = DescrStatsW(wt_avg_spin[j,:], weights=1/(wt_std_spin[j,:])**2, ddof=1).std
        
    return(KO_weighted_avg, KO_weighted_std, wt_weighted_avg, wt_weighted_std, KO_spins)

def KO_diff_sim(KO_avg, KO_std ,wt_avg, wt_std, thr_significance=3):
    """ Compute the differences between the average activity of the KO and the WT
    Args:
        - KO_avg (array): average activity of the KO
        - KO_std (array): standard deviation of the KO
        - wt_avg (array): average activity of the WT
        - wt_std (array): standard deviation of the WT
        
    Output:
        - diff_sim (array): difference between the average activity of the KO and the WT
        - diff_sim_std (array): standard deviation of the difference
    """
    diff_sim = KO_avg - wt_avg
    diff_sim_std = np.sqrt(KO_std**2 + wt_std**2)
    not_significant = []
    for i in range(len(diff_sim)):
        # print(i, np.abs(diff_sim[i])- thr_significance*diff_sim_std[i])
        if np.abs(diff_sim[i])<thr_significance*diff_sim_std[i]:
            # print("The difference between the average activity of the KO and the WT is not significant for gene ", i)
            not_significant.append(i)
    # diff_sim_std = np.sqrt(KO_std**2 + KO_std**2)
    # logFC between KO and WT
    # diff_sim = np.log2(KO_avg/ wt_avg)
    # diff_sim_std = np.sqrt((wt_avg/(KO_avg*np.log(2)))**2 * (KO_std**2+wt_std**2))
    return(diff_sim, diff_sim_std, np.array(not_significant))

def KO_diff_ExpVsSim(logFC_Exp, diff_Sim, diff_Sim_std, genes_order = genes_order, thr_significance=3):
    """ Compute the agreement between the logFC of the experiment and the KO-WT difference for simulated data.
    Args:
        - logFC_Exp (array): logFC of the experiment
        - diff_Sim (array): difference between the average activity of the KO and the WT
        - diff_Sim_std (array): standard deviation of the difference
        - genes_order (list of strings): list of genes in the order of the interaction matrix (remember to remove the KO genes)
        - thr_significance (float): number of standard deviation to consider the difference significant
        
    Output:
        - in_agreement: fractions of considered genes (only significant data) in agreement between the experiment and the simulation
        - data_considered: number of considered genes (only significant LogFC and KO-WT difference)
        - idx_Acc: indexes of the genes that are significant for the LofFC of the experimental data and for the KO-WT difference    
    """
    comparison= np.array(np.sign(logFC_Exp)*np.sign(diff_Sim))
    index_logFC_Exp = np.where(logFC_Exp==0)[0]
    index_diffSim = np.where((np.abs(diff_Sim))<thr_significance*diff_Sim_std)[0]
    print(index_diffSim)
    print("KO_std-wt_std not significant for gene ", genes_order[index_diffSim], index_diffSim)

    # union of the two indexes
    idx_notAcc = np.union1d(index_logFC_Exp, index_diffSim)
    # find the indexes of the genes that are not in idx_notAcc
    idx_Acc = np.setdiff1d(np.arange(len(logFC_Exp)), idx_notAcc)
    data_considered = len(idx_Acc)
    
    # consider the comparison elements that are not in idx_notAcc
    comparison_sel = comparison[idx_Acc]
    
    if data_considered == 0:
        in_agreement = 0
        no_agreement = 1
    else:
        in_agreement = len(np.where(comparison_sel==1)[0])/data_considered
        no_agreement = len(np.where(comparison_sel==-1)[0])/data_considered
    
    # Check
    check_sum = in_agreement+no_agreement-1
    check = np.where(check_sum>0.001)[0]
    if check.size > 0:
        print("Error in comparison Exp and Sim")
        
    return(in_agreement, data_considered, genes_order[idx_Acc])

def KO_plof_Diff_LogFC(logFC, diff, diff_std, KO_genes_order, idx_notS, title, n_sigma=1):
    """_summary_

    Args:
        logFC (array): logFC for experimental data
        diff (array): difference between WT and KO for simulated data
        diff_std (array): standard deviation of the difference between WT and KO for simulated data
        KO_genes_order (array): genes order with the KO gene removed
        idx_notS (array): array of indexes of genes that are not significant
        title (str): title of the plot
    """
    x_range = np.arange(0, len(diff))
    plt.figure(figsize=(10,5))
    
    # simulated data
    plt.errorbar(x_range, diff, yerr=n_sigma*diff_std, fmt='o', color='slateblue', ecolor='slateblue', elinewidth=2, capsize=0, label='Simulated')
    # plot with a different color the genes that are not significant (idx_notS)
    if len(idx_notS)>0:
        plt.errorbar(x_range[idx_notS], diff[idx_notS], yerr=n_sigma*diff_std[idx_notS], fmt='o', color='red', ecolor='red', elinewidth=3, capsize=0, label='Not significant difference')

    # experimental data
    plt.plot(x_range, logFC, 'o', color='Lightseagreen', label='Experimental')
    
    # line at 0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # plot settings
    plt.xticks(x_range, KO_genes_order, rotation='vertical')
    plt.grid(which='both', axis='both')
    plt.title(title)
    plt.legend(fontsize=18)


    plt.show()

# -------------------------------------------------------------------------------------------------
# ------------------------------------- (For 3 KO genes) ------------------------------------------
# -------------------------------------------------------------------------------------------------

def KO_activity_sim(matx, field, genes_order, model, N_test_KO=100, n_time=9547):
    """(For 3 KO genes)
    Compute the activity of the simulated data N_test_KO times 
    args:
    - matx(numpy array): interaction matrix
    - field(numpy array): field values
    - genes_order(list of strings): list of genes in the order of the interaction matrix
    - model(): model using ml_wrapper
    - N_test_KO(int): number of times the simulation is performed
    
    output: 
    - KO_spins(numpy array): activity of the simulated data, array of size (n_genes, n_spins, n_sim)
    """
    KO_spins = np.zeros((len(genes_order), n_time, N_test_KO))

    for i in range(N_test_KO):
        np.random.seed(i*5)
        KO_spins[:,:,i] = model.generate_samples_SetData(matx=matx, field=field, seed=i*5)
    return(KO_spins)

def KO3_avg_weighted(ko, wt_spins, N_test_KO=100):
    """ Compute the average spins value for each gene in KO and in WT
    Args:
    - ko(numpy array): activity of the simulated data, array of size (n_genes, n_times, n_sim)
    - wt(numpy array): activity of the wild type data, array of size (n_genes, n_times)
    - N_test_KO(int): number of times the simulation is performed
        
    """
    # # average activity for each gene in WT
    # wt_avg = np.array(wt.mean(axis=1))
    # wt_std = np.array(wt.std(axis=1, ddof=1))/np.sqrt(wt.shape[1])
    
    # activity for each gene in WT - simulated
    wt_avg_spin = np.array(wt_spins.mean(axis=1))
    wt_std_spin = np.array(wt_spins.std(axis=1, ddof=1))/np.sqrt(wt_spins.shape[1])
    wt_weighted_avg = np.zeros(wt_avg_spin.shape[0])
    wt_weighted_std = np.zeros(wt_avg_spin.shape[0])    
    for j in range(wt_avg_spin.shape[0]):
        wt_weighted_avg[j] = DescrStatsW(wt_avg_spin[j,:], weights=1/(wt_std_spin[j,:])**2, ddof=1).mean
        wt_weighted_std[j] = DescrStatsW(wt_avg_spin[j,:], weights=1/(wt_std_spin[j,:])**2, ddof=1).std
    
    # activity for each gene in KO
    KO_avg_spin = np.zeros((ko.shape[0], N_test_KO))
    KO_std_spin = np.zeros((ko.shape[0], N_test_KO))
    print(KO_avg_spin.shape)
    KO_spins = ko + 1
    
    for i in range(N_test_KO):
        KO_avg_spin[:,i] = np.array(KO_spins[:,:,i].mean(axis=1))
        KO_std_spin[:,i] = np.array(KO_spins[:,:,i].std(axis=1, ddof=1))/np.sqrt(KO_spins[:,:,i].shape[1])
    
    KO_weighted_avg = np.zeros(KO_avg_spin.shape[0])
    KO_weighted_std = np.zeros(KO_avg_spin.shape[0])    
    for j in range(KO_avg_spin.shape[0]):
        KO_weighted_avg[j] = DescrStatsW(KO_avg_spin[j,:], weights=1/(KO_std_spin[j,:])**2, ddof=1).mean
        KO_weighted_std[j] = DescrStatsW(KO_avg_spin[j,:], weights=1/(KO_std_spin[j,:])**2, ddof=1).std
        
    return(KO_weighted_avg, KO_weighted_std, wt_weighted_avg, wt_weighted_std)


def KO_plots_SimMultiple(ko_avg, ko_std, wt_avg, wt_std, KO_genes_order):
    """(For 3 KO genes)
    plot the average activity of the simulated data and the original data"""
    plt.figure(figsize=(18,5))
    plt.errorbar(KO_genes_order, ko_avg, yerr=ko_std,  
                    alpha=1, 
                    fmt="o", ms = 10,
                    elinewidth=3,
                    color="steelblue",
                    capsize=10,
                    label= "simulated Data")

    plt.errorbar(KO_genes_order, wt_avg, yerr=wt_std,
                    alpha=1, 
                    fmt="o", ms = 10,
                    elinewidth=1,
                    color="indianred",
                    capsize=10,
                    label = "original data")
    plt.legend(loc="upper left", fontsize=16)
    plt.xticks(fontsize=12)
    plt.ylabel("Average spin", fontsize=16)
    plt.xlabel("Genes", fontsize=16)
    plt.title("Average spin values for each genes", fontsize=20)
    plt.grid(True)
    plt.show()
    
# --------------------------------------------------------------------------------------------
# ------------------------------- KO plots - SCODE -------------------------------------------
# --------------------------------------------------------------------------------------------
    
def KO_plots_SimMultiple_SCODE(KO_spins, KO_genes_order, wt_avg, wt_std):
    """(For 3 KO genes)
    compute the average and std of the activity of the simulated data"""
    # mean active time
    std_temp = KO_spins.reshape((KO_spins.shape[0],KO_spins.shape[1]*KO_spins.shape[2]))
    # KO_std_spin = np.array(KO_spins_std.mean(axis=1))
    KO_std_spin = std_temp.std(axis=1)
    KO_avg_spin = np.array(KO_spins.mean(axis=1))
    KO_avg_spin = np.array(KO_avg_spin.mean(axis=1))

    plt.figure(figsize=(18,5))
    plt.errorbar(KO_genes_order, KO_avg_spin, yerr=KO_std_spin/np.sqrt(len(wt_std)),  
                    alpha=1, 
                    fmt="o", ms = 10,
                    elinewidth=3,
                    color="steelblue",
                    capsize=10,
                    label= "simulated Data")

    plt.errorbar(KO_genes_order, wt_avg, yerr=wt_std/np.sqrt(len(wt_std)), 
                    alpha=1, 
                    fmt="o", ms = 10,
                    elinewidth=1,
                    color="indianred",
                    capsize=10,
                    label = "original data")
    plt.legend(loc="upper left", fontsize=16)
    plt.xticks(fontsize=12)
    plt.ylabel("Average spin", fontsize=16)
    plt.xlabel("Genes", fontsize=16)
    plt.title("Average spin values for each genes", fontsize=20)
    plt.grid(True)
    plt.show()
    
def WT_avg_w( wt_spins):
    """ Compute the weighted average of the activity of the WT genes
    Args:
        - wt_spins: array of shape (N_genes, N_time, N_test)
    """
    # activity for each gene in WT
    wt_avg_spin = np.array(wt_spins.mean(axis=1))
    wt_std_spin = np.array(wt_spins.std(axis=1, ddof=1))/np.sqrt(wt_spins.shape[1])
    wt_weighted_avg = np.zeros(wt_avg_spin.shape[0])
    wt_weighted_std = np.zeros(wt_avg_spin.shape[0])    
    for j in range(wt_avg_spin.shape[0]):
        wt_weighted_avg[j] = DescrStatsW(wt_avg_spin[j,:], weights=1/(wt_std_spin[j,:])**2, ddof=1).mean
        wt_weighted_std[j] = DescrStatsW(wt_avg_spin[j,:], weights=1/(wt_std_spin[j,:])**2, ddof=1).std
        
    return(wt_weighted_avg, wt_weighted_std)

def KO_plots_avgAct_SCODE(KO_avg, KO_std, wt_avg, wt_std, KO_genes_order, N_sigma=1):
    """ For SCODE
    Args: 
        - KO_avg: array of shape (N_genes)
        - KO_std: array of shape (N_genes)
        - wt_avg: array of shape (N_genes)
        - wt_std: array of shape (N_genes)
        - KO_genes_order: array of shape (N_genes) containing the genes names
    """
    plt.figure(figsize=(18,5))
    plt.errorbar(KO_genes_order, KO_avg, yerr=N_sigma*KO_std,  
                    alpha=1, 
                    fmt="o", ms = 10,
                    elinewidth=3,
                    color="steelblue",
                    capsize=10,
                    label= "Simulated KO")

    plt.errorbar(KO_genes_order, wt_avg, yerr=N_sigma*wt_std, 
                    alpha=1, 
                    fmt="o", ms = 10,
                    elinewidth=1,
                    color="indianred",
                    capsize=10,
                    label = "Simulated WT")
    plt.legend(loc="upper left", fontsize=16)
    plt.xticks(fontsize=12)
    plt.ylabel("Average GE", fontsize=16)
    plt.xlabel("Genes", fontsize=16)
    plt.title("Average GE values for each genes", fontsize=20)
    plt.grid(True)
    plt.show()