import pandas as pd
import numpy as np

def mov_avg(vect, wind=5):
    """
    Function to compute the OVERLAPPING average between #columns=wind
    INPUTS:
    - vect: the  matrix, numpy array
    - wind: windows size, int
    OUTPUT:
    - out_vect: vector with dim=dim(vect) where each columns is given by the average of #samples=wind selected preserving the time (sampling by keeping separated samples of different times)
    """
    np.random.seed(123)
    index_steps = np.loadtxt("/Users/cleliacorridori/Dropbox_2021 Dropbox/Jorah Mormont/Code/Work/GE_data/time_sep.txt", dtype=np.int32)

    out_vect00 = np.zeros((np.shape(vect)[0], index_steps[0])) 
    out_vect06 = np.zeros((np.shape(vect)[0], index_steps[1])) 
    out_vect12 = np.zeros((np.shape(vect)[0], index_steps[2])) 
    out_vect24 = np.zeros((np.shape(vect)[0], index_steps[3])) 
    out_vect48 = np.zeros((np.shape(vect)[0], index_steps[4])) 
    
    for ii in range(index_steps[0]): 
        out_vect00[:,ii] = np.mean(vect[:, np.random.choice(index_steps[0], size=wind, replace=False)], axis=1)
    for ii in range(index_steps[1]): 
        out_vect06[:,ii] = np.mean(vect[:, np.random.choice(np.arange(index_steps[0], np.sum(index_steps[:2])), size=wind, replace=False)], axis=1)
    for ii in range(index_steps[2]): 
        out_vect12[:,ii] = np.mean(vect[:, np.random.choice(np.arange(np.sum(index_steps[:2]), np.sum(index_steps[:3])), size=wind, replace=False)], axis=1)
    for ii in range(index_steps[3]): 
        out_vect24[:,ii] = np.mean(vect[:, np.random.choice(np.arange(np.sum(index_steps[:3]), np.sum(index_steps[:4])), size=wind, replace=False)], axis=1)
    for ii in range(index_steps[4]): 
        out_vect48[:,ii] = np.mean(vect[:,np.random.choice(np.arange(np.sum(index_steps[:4]), np.sum(index_steps[:5])), size=wind, replace=False)], axis=1)
    
    out_vect = np.concatenate((out_vect00, out_vect06, out_vect12, out_vect24, out_vect48), axis=1)

    return(out_vect)


def mov_avgNO2(vect, wind=50):
    """
    Function to compute the NON OVERLAPPING average between #columns=wind
    INPUTS:
    - vect: the  matrix, numpy array
    - wind: windows size, int
    OUTPUT:
    - 
    """
    np.random.seed(123)
    index_steps = np.loadtxt("/Users/cleliacorridori/Dropbox (Dropbox_2021)/Code/Work/GE_data/time_sep.txt", dtype=np.int32)

    out_vect00 = np.zeros((np.shape(vect)[0], int(np.floor(index_steps[0]/wind)))) 
    out_vect06 = np.zeros((np.shape(vect)[0], int(np.floor(index_steps[1]/wind)))) 
    out_vect12 = np.zeros((np.shape(vect)[0], int(np.floor(index_steps[2]/wind)))) 
    out_vect24 = np.zeros((np.shape(vect)[0], int(np.floor(index_steps[3]/wind)))) 
    out_vect48 = np.zeros((np.shape(vect)[0], int(np.floor(index_steps[4]/wind)))) 
    
    for ii in range(int(np.floor(index_steps[0]/wind))): 
        size = ii*wind
        out_vect00[:,ii] = np.mean(vect[:, size:size+wind], axis=1)
    
    for ii in range(int(np.floor(index_steps[1]/wind))): 
        size = ii*wind
        out_vect06[:,ii] = np.mean(vect[:, index_steps[0]+size:index_steps[0]+size+wind], axis=1)
    
    for ii in range(int(np.floor(index_steps[2]/wind))): 
        size = ii*wind
        out_vect12[:,ii] = np.mean(vect[:, np.sum(index_steps[:2])+size:np.sum(index_steps[:2])+size+wind], axis=1)
    
    for ii in range(int(np.floor(index_steps[3]/wind))): 
        size = ii*wind
        out_vect24[:,ii] = np.mean(vect[:, np.sum(index_steps[:3])+size:np.sum(index_steps[:3])+size+wind], axis=1)
    
    for ii in range(int(np.floor(index_steps[4]/wind))): 
        size = ii*wind
        out_vect48[:,ii] = np.mean(vect[:, np.sum(index_steps[:4])+size:np.sum(index_steps[:4])+size+wind], axis=1)
    
    #print(out_vect00.shape, out_vect06.shape, out_vect12.shape, out_vect24.shape, out_vect48.shape)
    out_vect = np.concatenate((out_vect00, out_vect06, out_vect12, out_vect24, out_vect48), axis=1)
    return(out_vect)

def mov_avgNO(vect, wind=5):
    """
    Function to compute the NON OVERLAPPING average between #columns=wind
    INPUTS:
    - vect: the  matrix, numpy array
    - wind: windows size, int
    OUTPUT:
    - 
    """
    out_vect = np.zeros((np.shape(vect)[0],int(np.floor(np.shape(vect)[1]/wind))))
    for ii in range(int(np.floor(np.shape(vect)[1]/wind))):
        size = ii*wind
#         print(size, size+wind)
#         print(np.mean(vect[:, size:size+wind]))
        out_vect[:,ii] = np.mean(vect[:, size:size+wind], axis=1)
    return(out_vect)


