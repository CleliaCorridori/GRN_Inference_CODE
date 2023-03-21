import pandas as pd
import numpy as np
import lib.funcs_general as funcs_general
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score 
from scipy.stats import norm


import sys
sys.path.append('../')



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


# ----------------------------------------------------------------------------------------------
# ------------------------------------------  PEARSON ------------------------------------------
# ----------------------------------------------------------------------------------------------

def shuffle_dataframe(df, genes_names, interactions, N_test=500, Nbin=30, inferred_int_thr=0.02, method="Pearson"):
    """ Shuffle the dataframe and compute the Pearson correlation matrix.

    Args:
        - df (dataframe): dataframe with the gene expression values
        - genes_names (array): list of genes names
        - interactions (array): list of interactions
        - N_test (int): number of random shuffles
        - Nbin (int): number of bins for the histogram
        - MaxThr (float): threshold for the Pearson correlation values
       
    """
    N_rows = df.shape[0]
    N_cols = df.shape[1]

    # Melt the dataframe
    trial = df.melt(ignore_index=False)
    val_rnd = np.array(trial.iloc[:,1]) #gene expression values
    # Create an array of zeros to save the correlation matrices
    corr_matrices  = np.zeros((N_test,len(genes_names), len(genes_names)))
    TP_frac_rnd = np.zeros(N_test)
    info_int_rnd= np.zeros((N_test, 4, len(interactions)))

    # Loop over the number of random shuffles
    for ii in range(N_test):
        np.random.seed(ii)
        # Random reshuffle of the GE data
        np.random.shuffle(val_rnd)
        # Reshape as the original dataframe
        val_rnd = val_rnd.reshape(N_rows,N_cols)
        trial_long = pd.DataFrame(val_rnd, index= genes_names, columns= df.columns).set_index(genes_names) # reshaped dataframe setting the genes as index
        # Pearson matrix
        if method == "Pearson":
            corr_matr = np.corrcoef(trial_long)
            np.fill_diagonal(corr_matr, float("Nan")) # fill the diagonal with NaN
        elif method == "MaxEnt":
            corr_matr = -np.linalg.pinv(np.corrcoef(trial_long))
            # corr_matr = corr_matr/np.nanmax(np.abs(corr_matr))
            np.fill_diagonal(corr_matr, float("Nan")) # fill the diagonal with NaN
            
        # save all the correlation values
        corr_matrices[ii] = corr_matr

        TP_frac_rnd[ii], info_int_rnd[ii,:,:], _ = funcs_general.TP_plot(interactions, corr_matr, genes_names, 
                                                       inferred_int_thr=inferred_int_thr, Norm_Matx = False,
                                                       data_type=" Best model for lN PST MB data",
                                                       figplot=False, verbose=False, nbin=Nbin, Norm = False)
    return(TP_frac_rnd, info_int_rnd, corr_matrices)


def plot_TP_fraction(TP_frac_rnd, true_frac=0.67, text="PST+MB"):
    """ Plot the TP fraction for the shuffled data.
    
    Args:
        - TP_frac_rnd (array, Ntest elements): array containing the TP fraction for the shuffled data
        - text (str): text to add to the title of the plot
    Output:
        - quantiles (array, 3 elements): quantiles (5 and 95) of the TP fraction
    """
    # Plot options
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)
    
    fig, ax = plt.subplots(1,1, figsize=(8,4))
    # Plot the fraction of correctly inferred interactions
    ax.plot(TP_frac_rnd.flatten(), 'o')

    # Add labels and title
    ax.set_xlabel('Iteration', fontsize=16)
    ax.set_ylabel('TP fraction', fontsize=16)
    ax.set_title('TP inferred', fontsize=16)
    ax.set_ylim([0,1])
    ax.axhline(0.67, color="darkred")
    plt.show()


    sns.set(style="darkgrid")
    # creating the figure containing the distribution and the box plot
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=(9, 6))
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)
    # assigning a graph to each ax
    sns.boxplot(TP_frac_rnd.flatten(), ax=ax_box)
    sns.histplot(data = TP_frac_rnd.flatten(), ax=ax_hist, stat="density", bins=19, binrange = (0,1))
    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')
    ax_hist.set(xlabel='fraction of correct interactions')
    plt.axvline(x = true_frac, color = 'red', label = 'axvline - full height')
    plt.title(text+" data", fontsize=18)
    plt.show()
    sns.set_style("whitegrid")
    plt.show()
    
    return(np.quantile(TP_frac_rnd.flatten(),[0.05, 0.5, 0.95]))

# ------------------------------  Interactions Studies ------------------------------

def fit_normal_distribution(data, noise_thr=3, text="", Nbins=19):
    """ Fit a normal distribution to the data and plot the histogram and the pdf.
    Args:
        - data (array): array containing the data
        - noise_thr (float): threshold for the noise
        - text (str): text to add to the title of the plot
    
    Output:
        - mu (float): mean of the fitted distribution
        - std (float): standard deviation of the fitted distribution
        - Nnoise (int): number of values above the noise threshold
        - Ntot (int): total number of values
        """
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)
    data = np.array([x for x in data.flatten() if ~np.isnan(x)])
    mu, std = norm.fit(data)

    # Plot the histogram
    plt.figure(figsize=(7,5))
    n, _, _ = plt.hist(data, bins=Nbins, density=True, alpha=0.6, color='darkblue')

    # Plot the PDF
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, Nbins)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=1.5, label = 'fit')
    plt.xlabel('Correlation', fontsize=16)
    plt.ylabel('Pdf', fontsize=16 )
    # title = ", mu = %.2f,  2std = %.2f" % (np.abs(mu), noise_thr*std)
    title = ", mu = %.2f,  %d$\sigma$ = %.2f" % (np.abs(mu), noise_thr, noise_thr*std)
    plt.title(text+title, fontsize=20)

    #evaluation of the fit
    centroids = (n[1:]+n[:-1])/2
    R_square = r2_score(n, p) 
    print('regression score', R_square) 
    print('The noise-threshold is ', np.round(std*noise_thr,3)) 

    plt.axvline(x = -noise_thr*std, color = 'r')
    plt.axvline(x = noise_thr*std, color = 'r', label = str(noise_thr)+' $\sigma$')
    plt.legend()
    plt.text(-std*noise_thr, np.max(n)-0.1*np.max(n), '$R^2$ = '+str(np.round(R_square,2)),
            bbox={'facecolor': 'b', 'alpha': 0.5, 'pad': 5})
    plt.show()
    return(std*noise_thr)

# ----------------------------------------------------------------------------------------------
# ------------------------------------------  MaxEnt  ------------------------------------------
# ----------------------------------------------------------------------------------------------