import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib
from numba import njit, prange
from scipy.optimize import curve_fit


@njit
def log_fire(firing, threshold=10):
    logic_fire = np.zeros(len(firing))
    for i in range(0, len(firing)):
        if firing[i] > threshold:
            logic_fire[i]=1
        else:
            logic_fire[i]=0
    return logic_fire

@njit
def trial_len(fires, thresh=15):
    lenUP=[]
    lenDW=[]
    up = 0
    down = 0
    
    check = 0
    checkU = 0
    checkD = 0
        
    for i in range(len(fires)-1):
        if (fires[i]==0 and fires[i+1]==0):
            down += 1
            check += 1
            checkD += 1
            
        if (fires[i]==0 and fires[i+1]==1):
            up += 1
            lenDW.append(down)
            down = 0
            check += 1
            checkU += 1
            
        if (fires[i]==1 and fires[i+1]==0):
            down = 1
            lenUP.append(up)
            up = 0
            check += 1
            checkD += 1
            
        if (fires[i]==1 and fires[i+1]==1):
            up += 1
            check += 1
            checkU += 1
            
        if (i == len(fires)-2):
            lenUP.append(up)
            lenDW.append(down)
    return(lenUP, lenDW, check, checkU, checkD)

# exponential fit
def func_fit(x, a,b,c):
    return a * np.exp(-b * x) + c

@njit
def firing_comp(delta, dt, spins):
    '''
    delta: time interval over which we compute the firing rate, in seconds;
    dt: considered time step length in seconds;
    spins: matrix of (spins)x(time steps), each element is equal to -1 or +1.
    '''
    int_t = int(delta/dt) # time interval number of steps
    nspikes_int = np.zeros(int_t)
    firing_rate = []
    for idx_t in range(spins.shape[1]):
        if idx_t<int_t:
            nspikes_int[idx_t] = np.sum(spins[:,idx_t] == 1)
        if idx_t>=int_t:
            nspikes_int = np.delete(nspikes_int,0)
            nspikes_int = np.append(nspikes_int,np.sum(spins[:,idx_t] == 1))

        if idx_t>=int_t-1:
            firing_rate.append(nspikes_int.sum()/spins.shape[0]/delta)
    return firing_rate


def remove_shortint(UP_state, dt = 1/30, threshold = 0.2):
    # dt set to 0.1 ms
    # threshold set to 200 ms
    UP_state_rem=[]
    for i in range(len(UP_state)):
        if UP_state[i]<(threshold/dt):
            UP_state_rem.append(i)

    UP_state = np.delete(UP_state,UP_state_rem)
    return UP_state


def UP_anal_tot(spins, dt=1/30, delta=0.3, tresh=10 ):
    '''
    delta: is the lenght of the window that we consider to compute the rate
    dt: time step length
    '''
    
    firing_rate = firing_comp(delta, dt, spins)
    log_UP = log_fire(firing_rate, threshold=tresh)
    UP_state, DOWN_state, *_ = trial_len(log_UP, thresh=15)
    UP_state=remove_shortint(UP_state)
    
    UP_state_time = np.array(UP_state)*dt
    DOWN_state_time = np.array(DOWN_state)*dt
    
    mean_timeU = np.mean(UP_state_time)
    std_timeU = np.std(UP_state_time)
    mean_timeD = np.mean(DOWN_state_time)
    std_timeD = np.std(DOWN_state_time)
    

    return (UP_state_time,DOWN_state_time, mean_timeU, mean_timeD, std_timeU, std_timeD)