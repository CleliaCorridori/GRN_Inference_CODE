import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from scipy import stats
# from scipy.optimize import curve_fit
# from scipy.stats import norm
# from scipy.stats import spearmanr
# import networkx as nx
# from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import os
from os import system
import sys
sys.path.append('../')

import lib.figs_funcs as figfunc

matplotlib.rc('text', usetex=True)
sns.set(font='Avenir')
sns.set(style="white")

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

# ------------------------------ average gene expression ------------------------------
def average_ge_in_time(df, time_sep, genes_type_name, genes_type, title, plot_avg=True):
    """ Compute the average expression of the different gropus of genes in the different time steps and plot the result
    Args: 
        df (dataframe): dataframe with the gene expression
        time_sep (list of ints): list with the number of time steps for each time point
        genes_type_name (list of strings): list with the name of the groups of genes
        genes_type (list of lists): list of lists of genes for each group
        title (string): title of the plot
        plot_avg (bool): if True returns the plot 
    """
    df_bulk = pd.concat([df.iloc[:,:time_sep[0]].mean(axis=1), df.iloc[:, time_sep[0]: np.sum(time_sep[:2])].mean(axis=1), df.iloc[:, np.sum(time_sep[:2]): np.sum(time_sep[:3])].mean(axis=1), df.iloc[:, np.sum(time_sep[:3]): np.sum(time_sep[:4])].mean(axis=1), df.iloc[:, np.sum(time_sep[:4]): np.sum(time_sep[:5])].mean(axis=1)], 
                        axis=1)
    df_bulk.columns=["00", "06","12", "24", "48"]
    df_bulk_groups = pd.DataFrame(index=genes_type_name, columns=df_bulk.columns)
    for ii in range(len(genes_type_name)):
        df_bulk_groups.iloc[ii,:] = df_bulk.loc[genes_type[ii]].mean(axis=0)
    # compute the std for each time step for each group of genes (naive, formative, committed)
    df_std = pd.concat([df.iloc[:,:time_sep[0]].std(axis=1), df.iloc[:, time_sep[0]: np.sum(time_sep[:2])].std(axis=1), df.iloc[:, np.sum(time_sep[:2]): np.sum(time_sep[:3])].std(axis=1), df.iloc[:, np.sum(time_sep[:3]): np.sum(time_sep[:4])].std(axis=1), df.iloc[:, np.sum(time_sep[:4]): np.sum(time_sep[:5])].std(axis=1)], axis=1)
    df_std.columns=["00", "06","12", "24", "48"]
    df_std_groups = pd.DataFrame(index=genes_type_name, columns=df_bulk.columns)
    df_std_groups.columns=df_bulk.columns
    
    for ii in range(len(genes_type_name)):
        df_std_groups.iloc[ii,:] = df_std.loc[genes_type[ii]].mean(axis=0)
    
    if plot_avg:
        plt.figure(figsize=(12,8))
        plt.errorbar(df_bulk_groups.columns, df_bulk_groups.iloc[0,:], yerr=df_std_groups.iloc[0,:], color='darkblue', label='naive', marker='o')
        plt.errorbar(df_bulk_groups.columns, df_bulk_groups.iloc[1,:], yerr=df_std_groups.iloc[1,:], color='darkgreen', label='formative', marker='o')
        plt.errorbar(df_bulk_groups.columns, df_bulk_groups.iloc[2,:], yerr=df_std_groups.iloc[2,:], color='darkred', label='committed', marker='o')
        plt.legend(fontsize=15)
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('Average expression', fontsize=15)
        plt.title('Average expression per group', fontsize=18)
        plt.grid()
        plt.ylim([0,1.75])
        plt.title(title, fontsize=18)
        plt.show()
    return(df_bulk_groups, df_std_groups, df_bulk, df_std)

def plot_ge_in_time(df_bulk, df_std, genes_type, title):
    """Plot the gene expression for each gene in each group in time with theis std

    Args:
        df_bulk (dataframe): average gene expression for each gene in each group in time
        df_std (dataframe): std of the gene expression for each gene in each group in time
        genes_type (list of lists): list of lists of genes for each group
        title (list of strings): list of strings with the title for each group
    """
    for ii in range(len(genes_type)):
        plt.figure(figsize=(12,8))
        for jj in range(len(genes_type[ii])):
            plt.errorbar(df_bulk.columns,df_bulk.loc[genes_type[ii][jj]].T, yerr=df_std.loc[genes_type[ii][jj]].T, label=genes_type[ii][jj], marker='o')
        plt.legend(fontsize=16)
        plt.title(title[ii], fontsize=20)
        plt.grid()
        plt.show()

# ------------------------------ binnarization ----------------------------------------
def binnarization(df, thr=0.5, genes_order=[]):
    """ Binarize the values of a DataFrame based on a threshold
    input: 
        df: DataFrame to binarize
        thr: threshold to binarize the values
        genes_order: list of genes in the same order as the rows of the DataFrame
        
    output:
        spins_df: binarized DataFrame
    """
    # Calculate the maximum value for each row of the DataFrame
    df_max = df.max(axis=1)

    # Create a copy of the DataFrame to store the binarized values
    spins_df = df.copy()

    # Loop through each row of the DataFrame: genes_order contains the genes of the dataframe
    for ii in range(len(genes_order)):
        # Binarize the values in the row
        spins_df.iloc[ii,:] = (df.iloc[ii,:] > (df_max[ii] * thr)).astype(float)

    # Replace all 0 values with -1
    spins_df[spins_df==0] = -1

    # Return the binarized DataFrame
    return spins_df


# ------------------------------ randomization
def check_shuffle(dataset, Ntest):
    """Function to check if the randomization is working properly.
    input:
    dataset: dataset to check, dimension: (Ntest, Ngenes, Ncells)
    Ntest: number of tests performed
    output:
    percentage of equal elements
    """
    # Check: for each test the result should be different
    check_eq = 0
    for ii in range(Ntest):
        for jj in range(ii+1, Ntest):
            # number of equal elements
            temp = len(np.where(np.abs(dataset[ii,:,:]-dataset[jj,:,:])<0.001)[0]) # number of equal elements
            if temp > 0.9*dataset.shape[1]*dataset.shape[2]: # if more than 70% of the elements are equal
                check_eq += 1 # count the number of equal elements
    return(check_eq/(Ntest*(Ntest-1))*2) # percentage of equal elements


# ------------------------------ check correct interactions given the adjacency matrix
def TP_check(interaction_list, interaction_matrix, genes_list, inferred_int_thr = 0, Norm = True):
    """
    NOTE: for not symmetric interaction_matrix:
    - rows: who undergoes the action;
    -columns: who acts.
    NOTE: to read the input interaction_list:
    - couple[0] : who acts;
    - couple[1] : who undergoes the action.
    
    NOTE: inferred_int_thr is computed as fraction of max(interaction_matrix) 
    
    inputs: 
    interaction_list: list of strings, each string is a couple of genes and the interaction value (+1, -1 )
    interaction_matrix: matrix of interactions
    genes_list: list of genes names
    inferred_int_thr: threshold to consider an interaction as inferred
    Norm: if True, normalize the known interaction divding by the maximum value of the interaction_matrix
    
    output:
    out_matx: matrix of interactions, dimension: (4, len(interaction_list)). The rows are:
            - row 0: who acts;
            - row 1: who undergoes the action;
            - row 2: interaction value;
            - row 3: 1 if the interaction is correctly inferred, 0 otherwise.
    """
    if Norm:
        m_max = np.nanmax(np.abs(interaction_matrix))
    else:
        m_max = 1
    
    # true_positive=0
    # int_val = np.zeros(len(interaction_list))
    # wrong_ints = []
    out_matx = np.zeros((4, len(interaction_list)))
    
    for ii in range(len(interaction_list)): # split the list of strings
        couple = interaction_list[ii].split(" ")
        gene1_idx = np.where(genes_list == couple[1])[0] #idx of gene 1
        gene0_idx = np.where(genes_list == couple[0])[0] #idx of gene 0  
        
        # check if the interaction's genes already exist:
        if (len(np.where(genes_list == couple[0])[0])==0):
            print("Gene "+ couple[0]+" not found")
            continue
        if (len(np.where(genes_list == couple[1])[0])==0):
            print("Gene "+ couple[1]+" not found")
            continue
            
        # the subjects of the interaction
        out_matx[0,ii] = gene0_idx # who acts
        out_matx[1,ii] = gene1_idx # who acts
  
        # the interaction value (and the sign of the interaction)
        out_matx[2,ii] = interaction_matrix[gene1_idx[0], gene0_idx[0]]
        interaction = np.sign(out_matx[2,ii])
#         print("CHECK", len(np.where(np.abs(out_matx[2,ii]) >= inferred_int_thr*m_max)))
#         print("check: ", np.abs(out_matx[2,ii]), "\n", inferred_int_thr*m_max)
        if (interaction==int(couple[2])) and (np.abs(out_matx[2,ii])/m_max >= inferred_int_thr):
            out_matx[3,ii] = 1
        elif  (interaction==int(couple[2])) and (int(couple[2])==0):
            out_matx[3,ii] = 1
        else:
            out_matx[3,ii] = 0

    return(np.sum(out_matx[3,:])/len(out_matx[3,:]), out_matx)
        
        
def TP_plot(interaction_list, interaction_matrix, genes_order, inferred_int_thr=0, Norm_Matx = False,
            data_type="scRNA-seq PST MB", 
            figplot=True, nbin=30, 
            verbose=False, Norm=True):
    """Wrap function to visualize all the results of the comparison with the known interactions (TP)
    input:
    interaction_list: list of known interactions,
    interaction_matrix: matrix of inferred interactions,
    genes_order: list of genes in the same order as the rows of the interaction_matrix,
    inferred_int_thr: threshold to consider an interaction as correctly inferred (otherwise it is 0),
    Norm_Matx: if True, normalize the interaction_matrix to the maximum value,
    data_type: string only to print the type of data,
    verbose: if True, print the fraction of true positives and the TP and all interaction values,
    Norm: if True, normalize the KNOWN interactions to the maximum value of the interaction_matrix
    
    output:
    TP_fraction: fraction of true positives,
    TP_info: matrix of interactions, dimension: (4, len(interaction_list)) -> see TP_check function "out_matx" output
    interaction_matrix: matrix of inferred interactions, useful if it is normalized.
    """
    
    if Norm_Matx:
        interaction_matrix = interaction_matrix/np.nanmax(np.abs(interaction_matrix))

    # Check the list of known interactions correctly inferred
    TP_fraction, TP_info = TP_check(interaction_list, interaction_matrix, genes_order, inferred_int_thr, Norm=Norm)
    
    # Print the fraction of true positives and the TP and all interaction values:
    if verbose==True:
        print("\nRESULTS for " + data_type)
        print("\nTP fraction:", np.round(TP_fraction, 2))
        print("\nInteraction values:\n", np.round(TP_info[2,:],3))
        print("\nTP ints values:\n", np.round(TP_info[2,:]*TP_info[3,:],3))
    
    # If the figplot flag is set to True, plot the matrix and the distribution of the INTERACTION MATRIX
    if figplot==True:
        bins = np.linspace(np.ndarray.flatten(interaction_matrix).min(), np.ndarray.flatten(interaction_matrix).max(), nbin)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
        figfunc.plotmat(interaction_matrix, fig, ax[0], genes_order, data_type+"")
        sns.histplot(np.ndarray.flatten(interaction_matrix), ax=ax[1], stat="density", bins=bins)
        plt.show()
        
    return(TP_fraction, TP_info, interaction_matrix)

# ----------------------- # BEST MODEL SELECTION using known interactions
def Inferred_matrix(prec_sel, matrices_sel, interactions, genes_order, interaction_thr, text):
    """Function to select the best inferred matrix from the grid search"""
    #select the matrices with higher fraction of correctly inferred interactions
    high_idx = np.where(prec_sel == prec_sel.max())[0] 
    high_matrices = matrices_sel[high_idx,:,:]

    high_info_int = np.zeros((4, len(interactions), len(high_idx)))
    high_TP_frac=np.zeros(len(high_idx))

    high_meanMatx = np.zeros((len(genes_order), len(genes_order)))
    counter = 0
    idxs = []
    for ii in range(len(high_idx)):
        high_TP_frac[ii], high_info_int[:,:,ii], matrix = TP_plot(interactions, high_matrices[ii,:,:], genes_order, 
                                                    inferred_int_thr=interaction_thr, Norm_Matx = False,
                                                    data_type= text + str(ii+1) +", "+ str(np.round(prec_sel[high_idx[ii]],2)),
                                                    figplot=False, verbose=False, nbin=30, Norm=True)
        # print(f"THR: {'{:.3f}'.format(np.max(np.abs(matrix))*interaction_thr)}")

        if high_TP_frac[ii] == prec_sel.max():
            high_meanMatx += high_matrices[ii,:,:]
            counter += 1
            idxs.append(high_idx[ii])
            
    high_meanMatx = high_meanMatx/counter 
    return(high_meanMatx, np.array(idxs))


# ----------------------- # LogFC INFO
def InteractionList(df, perc=0):
    """function to extract the list of interactions from a dataframe of logFC values in the format:
    list of strings
    each string: "gene1 gene2 sign"

    Args:
        df (dataFrame): dataframe of logFC values
        perc (float, optional): threshold to consider an interaction. Defaults to 0.
        
    output: list of interactions 
    """
    thr = np.abs(df.max().max()*perc)
    output = []
    for row in df.index:
        for col in df.columns:
            element = df.loc[row, col]
            if element > thr:
                sign = "1"
            elif element < -thr:
                sign = "-1"
            else:
                df.loc[row, col] = 0
                sign = "0"
            if (sign == "-1") or (sign == "1"):
                output.append(f"{col} {row} {sign}")
    return(output)
