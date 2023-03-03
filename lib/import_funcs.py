import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats

# --------------------
# ----- TO SAVE ------
# --------------------

def GE_matrix_out(path, text, index=True):
    """ Function to SAVE the data matrix, merging all the times
        it saves:
        - the matrix with all the times data;
        + it returns all the matrices at different time steps and the whole matrix.
        
        NOTE: THE FUNCTION IS SLOW because it loads the matrices from the folder, not inside the notebook """
    times_idx_barcodes_matx = np.load(path+"/general_info/cells_times/times_cells_indices.npz")
    
    df = pd.read_csv(path+'input_data/'+text+'_w00.csv', header=1, sep=",")
    matx = np.squeeze(np.array(df))[:,1:]
    matx = matx[:,1:]
    
    matx_00 = matx[:, times_idx_barcodes_matx['idx_00']]
    matx_06 = matx[:, times_idx_barcodes_matx['idx_06']]
    matx_12 = matx[:, times_idx_barcodes_matx['idx_12']]
    matx_24 = matx[:, times_idx_barcodes_matx['idx_24']]
    matx_48 = matx[:, times_idx_barcodes_matx['idx_48']]
    
    matx_clean = np.concatenate((matx_00, matx_06, matx_12, matx_24, matx_48), axis=1)
    np.savetxt(path+'/GE_matrices/'+text+'.csv', matx_clean, fmt='%s')
    
    if index==True:
        initial_idx = np.array([0, #00h
                            np.shape(matx_00)[1], #06h
                            np.shape(matx_00)[1]+np.shape(matx_06)[1], #12h
                            np.shape(matx_00)[1]+np.shape(matx_06)[1]+np.shape(matx_12)[1], #24h
                            np.shape(matx_00)[1]+np.shape(matx_06)[1]+np.shape(matx_12)[1]+np.shape(matx_24)[1]]) #48h

        final_idx   = np.array([np.shape(matx_00)[1]-1, #0h
                                np.shape(matx_00)[1]-1+np.shape(matx_06)[1], #6h
                                np.shape(matx_00)[1]-1+np.shape(matx_06)[1]+np.shape(matx_12)[1], #12h
                                np.shape(matx_00)[1]-1+np.shape(matx_06)[1]+np.shape(matx_12)[1]+np.shape(matx_24)[1], #24h
                                np.shape(matx_00)[1]-1+np.shape(matx_06)[1]+np.shape(matx_12)[1]+np.shape(matx_24)[1]+np.shape(matx_48)[1]]) 

        np.savetxt(path+'general_info/cells_times/initial_idx.csv', initial_idx, fmt='%s')
        np.savetxt(path+'general_info/cells_times/final_idx.csv', final_idx, fmt='%s')

    
    return(matx_00, matx_06, matx_12, matx_24, matx_48, matx_clean)


def GE_matrix_out_MostImp(path, text, save_matx=True):
    """Function to SAVE only specific rows of Gene Expression 
    - It saves the whole matrix with all the time steps, just for the 24 important genes
    + it returns also inside the notebook the matriz
    
    NOTE: THE FUNCTION IS SLOW because it loads the matrices from the folder, not inside the notebook """
    imp_genes = np.loadtxt(path+"general_info/imp_genes.csv", dtype="str")
    genes = np.loadtxt(path+'general_info/all_genes_list.csv', dtype="str")
    arr_index = np.zeros(len(imp_genes))
    for jj in range(len(imp_genes)):
        arr_index[jj] = np.where(genes == imp_genes[jj])[0]
    arr_index = arr_index.astype(int)
    
    matx = np.loadtxt(path+'/GE_matrices/'+text+'.csv')
    matx_imp_genes = matx[arr_index, :]
    
    if save_matx==True:
        np.savetxt(path+'/GE_matrices/'+text+'_impGenes.csv', matx_imp_genes, fmt='%s')
        print(text+": matrix saved")
    return(matx_imp_genes)


# --------------------
# ----- TO LOAD ------
# --------------------

def GE_time_matrices(matx, path, text, index=True):
    """ Function to LOAD the data matrix + divide it in submatrices at different time points
        and return all the matrices at different steps"""
    print("--elaborating "+text+"--")
    # matx = np.loadtxt(path+'/GE_matrices/'+text+'.csv') # Gene expression matx
    initial_idx = np.loadtxt(path+"general_info/cells_times/initial_idx.csv", dtype="int")
    final_idx = np.loadtxt(path+"general_info/cells_times/final_idx.csv", dtype="int")
    
    a = [[matx[:,initial_idx[ii]:final_idx[ii]+1]] for ii in range(len(initial_idx))]
    return(np.squeeze(np.array(a[0])), np.squeeze(np.array(a[1])), np.squeeze(np.array(a[2])), np.squeeze(np.array(a[3])),                  np.squeeze(np.array(a[4])))

def impGE_time_matrices(path, text, index=True):
    """ Function to LOAD the data matrix of the 24 most important genes + divide it in submatrices at different time points
        and return all the matrices at different steps"""
    print("--loading "+text+"--")
    matx = np.loadtxt(path+'/GE_matrices/'+text+'_impGenes.csv') # Gene expression matx
    initial_idx = np.loadtxt(path+"general_info/cells_times/initial_idx.csv", dtype="int")
    final_idx = np.loadtxt(path+"general_info/cells_times/final_idx.csv", dtype="int")
    
    a = [[matx[:,initial_idx[ii]:final_idx[ii]+1]] for ii in range(len(initial_idx))]
    return(np.squeeze(np.array(a[0])), np.squeeze(np.array(a[1])), np.squeeze(np.array(a[2])), np.squeeze(np.array(a[3])), np.squeeze(np.array(a[4])))

def split_time_matrices(matx, path):
    initial_idx = np.loadtxt(path+"general_info/cells_times/initial_idx.csv", dtype="int")
    final_idx = np.loadtxt(path+"general_info/cells_times/final_idx.csv", dtype="int")
    a = [[matx[:,initial_idx[ii]:final_idx[ii]+1]] for ii in range(len(initial_idx))]
    return(np.squeeze(np.array(a[0])), np.squeeze(np.array(a[1])), np.squeeze(np.array(a[2])), np.squeeze(np.array(a[3])), np.squeeze(np.array(a[4])))

def load_genes(path):
    """ This function loads the list of all used genes (=features)"""
    return(np.loadtxt(path+'general_info/all_genes_list.csv', dtype="str"))

def load_cells(path):
    """ This function loads the list of all used genes (=features)"""
    return(np.loadtxt(path+'general_info/all_cells_list.csv', dtype="str"))

# ---------------- BULK ----------------
def get_bulk_data(df, all_tog=False):
    col_names = np.array(df.columns)

    col_names_2iL_1 = [col_names[3+2*i] for i in range(8)]
    col_names_2iL_2 = [col_names[2+2*i] for i in range(8)]
    col_names_2i_1 = [col_names[18+2*i] for i in range(8)]
    col_names_2i_2 = [col_names[19+2*i] for i in range(8)]
 
    df_2iL_1 = np.array(df.loc[:, col_names_2iL_1])
    df_2iL_2 = np.array(df.loc[:, col_names_2iL_2] )

    df_2i_1 = np.array(df.loc[:, col_names_2i_1])
    df_2i_2 = np.array(df.loc[:, col_names_2i_2] )
    
    vect_2iL = np.concatenate((df_2iL_1,df_2iL_2), 1)
    vect_2i  = np.concatenate((df_2i_1,df_2i_2), 1)
    if all_tog==False: # return 2i+LIF and 2i separate
        return(vect_2iL, vect_2i)
    elif all_tog==True:
        return(np.concatenate((vect_2iL , vect_2i),1))
    elif all_tog == 2:
        df_2iL_Mean = (df_2iL_1+df_2iL_2)/2 
        df_2i_Mean = (df_2i_1+df_2i_2)/2 
        return(df_2iL_Mean, df_2i_Mean)