import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib
from numba import njit, prange
from lib.import_funcs import *

# genes 
#genes = load_genes(path)
#imp_genes = np.loadtxt(path+"general_info/imp_genes.csv", dtype="str")
#time=["00h", "06h", "12h", "24h", "48h"]

def dictionary_by_time(df, no_cols = True):
    path = '/Users/cleliacorridori/Dropbox (Dropbox_2021)/Clelia/Work/' # for Mac
    cells = load_cells(path)
    if no_cols==True:
        df = pd.DataFrame(df)
        df.columns = cells
    initial_idx = np.loadtxt(path+"general_info/cells_times/initial_idx.csv", dtype="int")
    final_idx = np.loadtxt(path+"general_info/cells_times/final_idx.csv", dtype="int")
    array= np.zeros(np.shape(df)[1])
    time_steps = [0, 6, 12, 24, 48]
    for ii in range(len(initial_idx)):
        array[initial_idx[ii]:final_idx[ii]+1]=time_steps[ii]
    array = np.reshape(array, [1,len(array)])
    array = pd.DataFrame(array, columns = cells).set_index(np.array(["Time"]))

    df = pd.concat([array, df])
    
    # SPLIT BY TIME
    data_dict = {"00" :df.loc[:, df.loc['Time'] == 0].drop(["Time"]),
                 "06" :df.loc[:, df.loc['Time'] == 6].drop(["Time"]),
                 "12" :df.loc[:, df.loc['Time'] == 12].drop(["Time"]),
                 "24" :df.loc[:, df.loc['Time'] == 24].drop(["Time"]),
                 "48" :df.loc[:, df.loc['Time'] == 48].drop(["Time"])}
    return(data_dict)
