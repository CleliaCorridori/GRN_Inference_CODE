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


# ------------------------------ Plot average activity time of genes

def plot_activity_simulated(spins_df_sim, genes_order, title, color, ax):
    # spins_df_sim is (n_genes, n_time, n_test)
    avg_activity     = spins_df_sim.mean(axis=1).mean(axis=1)
    avg_activity_std = spins_df_sim.std(axis=1).mean(axis=1)
    ax.errorbar(genes_order, avg_activity, yerr=avg_activity_std/np.sqrt(spins_df_sim.shape[0]), 
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
    avg_activity_std = spins_df.std(axis=1)
    ax.errorbar(genes_order, avg_activity, yerr=avg_activity_std/np.sqrt(spins_df.shape[0]), 
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

# ------------------------------ Distance between matrices

def sum_squared_abs_diff(array1, array2):
    """Calculate the sum of squared absolute differences between two matrices"""
    diff = (array1.flatten()-array2.flatten())**2
    return np.sqrt(np.sum(diff))

# ------------------------------ Knockout functions
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

def KO_plots_oneSim(model, matx, field, KO_genes_order, wt_avg, wt_std, seed=1, raster=True, avg=True):
    """function to simulate the KO data (active/inactive genes in time) and to plot, depending on the decision of the user, 
    the raster plot and the average active time for each gene in wild type and in KO.
    Args:
        model (): model using ml_wrapper
        matx (numpy array): interaction matrix
        field (numpy array): field values
        KO_genes_order (list of strings): list of genes in the order of the interaction matrix
        wt_avg (numpy array): average activity of the wild type genes
        wt_std (numpy array): average std of the wild type genes
        seed (int, optional): seed. Defaults to 1.
        raster (bool, optional): decide to plot the raster plot or not. Defaults to True.
        avg (bool, optional): decide to plot the average active time for each gene in wild type and in KO. Defaults to True.
    """
    #generate new data
    KO_spins = model.generate_samples_SetData(matx=matx, field= field, seed=seed)
    
    # raster plot
    if raster:
        fun_plotting.raster_plot(KO_spins, 'Reconstruction', 1, KO_genes_order)
        plt.show()
        
    # average activity time per gene in wild type and in KO
    if avg:
        # mean active time
        KO_std_spin = np.array(KO_spins.std(axis=1))
        KO_avg_spin = np.array(KO_spins.mean(axis=1))+1

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
    return(KO_spins)
        
def KO_plotLogFC_ExpAndSim(lofFC_Exp, logFC_Sim, KO_genes_order):
    """Plot the logFC of the experiment and the simulated data
    input:
    lofFC_Exp: logFC of the experiment, array of size (n_genes)
    logFC_Sim: logFC of the simulated data, array of size (n_genes, n_sim)
    KO_genes_order: order of the genes in the simulation and experiment, array of size (n_genes)
    """
    plt.figure(figsize=(18,5))
    plt.plot(lofFC_Exp, 
             "o",ms = 10, label="Exp")
    plt.plot(logFC_Sim[:,0],  
             "o", ms = 10,
             color="darkred", label= "Sim")
    plt.xticks(np.arange(0,23),KO_genes_order)
    plt.axhline(0)
    plt.legend()

    
def KO_logFC_sim(matx, field,genes_order,wt_avg, model, N_test_KO=100):
    """Compute LogFC for SIMULATED data
    input:
    matx(numpy array): interaction matrix
    field(numpy array): field values
    genes_order(list of strings): list of genes in the order of the interaction matrix
    wt_avg(numpy array): average activity of the wild type genes
    model(): model using ml_wrapper
    
    output:
    diff_sim(numpy array): logFC of the simulated data, array of size (n_genes, n_sim)
    """
    diff_sim = np.zeros((len(genes_order)-1, N_test_KO))
    for i in range(N_test_KO):
        KO_spins = model.generate_samples_SetData(matx=matx, field=field, seed=i*5)
        KO_avg_spin = np.array(KO_spins.mean(axis=1))+1
        KO_std_spin = np.array(KO_spins.std(axis=1))

        # Comparison
        diff_sim[:,i] = np.log2(KO_avg_spin/wt_avg)
    #     diff_sim = KO_pN_mb_pst_avg_spin-wt_pN_mb_pst_avg_spin
#         diff_sim_ii[np.abs(diff_sim_ii)<np.max(np.abs(diff_sim_ii))*thr_KO]=0    # set the threshold
#         diff_sim[:,i] = diff_sim_ii
    return(diff_sim)


def KO_comparison_ExpVsSim(lofFC_Exp, logFC_Sim, N_test=100):
    """compute the fraction of Experimental Data end Simulated data in Agreement
    lofFC_Exp: logFC of the experiment, array of size (n_genes)
    logFC_Sim: logFC of the simulated data, array of size (n_genes, n_sim)
    
    output:
    mean_in_agreement: mean fraction of Experimental Data end Simulated data in Agreement
    data_considered: number of genes considered in the comparison"""
    comparison= np.array([np.sign(lofFC_Exp)*np.sign(logFC_Sim[:,ii]) for ii in range(N_test)])
#     print(np.sum([len(np.where(logFC_Sim[:,ii]==0)[0]) for ii in range(N_test)]))
    data_considered = np.array([len(np.where(comparison[ii,:]!=0)[0]) for ii in range(N_test)])
    
    in_agreement = np.array([len(np.where(comparison[ii,:]==1)[0])/data_considered[ii] for ii in range(N_test)])
    no_agreement = np.array([len(np.where(comparison[ii,:]==-1)[0])/data_considered[ii] for ii in range(N_test)])
    mean_in_agreement = np.mean(in_agreement)
    
    # Check
    check_sum = np.array([in_agreement[ii]+no_agreement[ii] for ii in range(N_test)])-1
    check = np.where(check_sum>0.001)[0]
    if check.size > 0:
        print("Error in comparison Exp and Sim")
    return(mean_in_agreement,    data_considered)

# ---------------------------- (For 3 KO genes) --------------------------------
def KO_activity_sim(matx, field,genes_order, model, N_test_KO=100, n_time=9547):
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
    - KO_avg_spin(numpy array): average activity of the simulated data, array of size (n_genes, n_sim)
    - KO_std_spin(numpy array): std activity of the simulated data, array of size (n_genes, n_sim)
    """
    KO_spins = np.zeros((len(genes_order), n_time, N_test_KO))
    for i in range(N_test_KO):
        np.random.seed(i*5)
        KO_spins[:,:,i] = model.generate_samples_SetData(matx=matx, field=field, seed=i*5)
        KO_avg_spin = np.array(KO_spins.mean(axis=1))+1
        KO_std_spin = np.array(KO_spins.std(axis=1))

    return(KO_spins, KO_avg_spin, KO_std_spin)


def KO_plots_SimMultiple(KO_spins, KO_genes_order, wt_avg, wt_std):
    """(For 3 KO genes)
    compute the average and std of the activity of the simulated data"""
    # mean active time
    std_temp = KO_spins.reshape((KO_spins.shape[0],KO_spins.shape[1]*KO_spins.shape[2]))
    # KO_std_spin = np.array(KO_spins_std.mean(axis=1))
    KO_std_spin = std_temp.std(axis=1)
    KO_avg_spin = np.array(KO_spins.mean(axis=1))+1
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