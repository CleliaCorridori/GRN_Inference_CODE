import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt

class MidpointNormalize(pltcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        pltcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

    
def plotmat(m, fig, ax, ax_names, text, cmap = 'RdBu_r'):
    #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    img = ax.imshow(m, cmap = cmap, clim=(np.nanmin(m), np.nanmax(m)),
                    norm = MidpointNormalize(midpoint=0,
                                             vmin=np.nanmin(m),
                                             vmax=np.nanmax(m))
                   )
    
    cbarM = fig.colorbar(img, ax = ax)
    cbarM.set_label(r'$C_{ij}$', rotation = -90, labelpad = 20, fontsize = 12)
    cbarM.ax.tick_params(labelsize = 12)

    ax.set_xlabel('Gene label', fontsize = 18, labelpad = 10)
    ax.set_ylabel('Genes label', fontsize = 18, labelpad = 10)
    ax.set_xticks(np.arange(0,np.shape(np.array(m))[0]))
    ax.set_xticklabels(ax_names, rotation='vertical', fontsize=16)
    ax.set_yticks(np.arange(0,np.shape(np.array(m))[0]))
    ax.set_yticklabels(ax_names, fontsize=16)
    ax.set_title(text, fontsize = 20)

    #ax.title.set_text(text)

    #return(ax)
    #plt.show()