import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import networkx as nx
import seaborn as sns
import sys
sys.path.append('../')
import seaborn as sns
from matplotlib.lines import Line2D


from lib.import_funcs import *
import lib.figs_funcs as figfunc

# ------------------------------

path = "/Users/cleliacorridori/Dropbox_2021 Dropbox/Jorah Mormont/GRN_Inference/DATA/" # for Mac

# genes of OUR dataset
genes = np.loadtxt(path+'general_info/all_genes_list.csv', dtype="str")
cells = np.loadtxt(path+'general_info/all_cells_list.csv', dtype="str")
imp_genes = np.loadtxt(path+"general_info/imp_genes.csv", dtype="str") #selected genes

# time steps
time=["00h", "06h", "12h", "24h", "48h"]

# Genes Classification
naive = ["Klf4", "Klf2", "Esrrb", "Tfcp2l1", "Tbx3", "Stat3", "Nanog", "Sox2"]
formative = ["Nr0b1", "Zic3", "Rbpj", "Utf1", "Etv4", "Tcf15"]
committed = ["Dnmt3a", "Dnmt3b", "Lef1", "Otx2", "Pou3f1", "Etv5"]

# Selected genes order
genes_order = np.concatenate((naive, formative, committed))
nc_genes = np.setdiff1d(imp_genes, genes_order)
genes_order = np.concatenate((genes_order, nc_genes))

# ------ INTERACTION MATRIX ------

def intM_plot(matx, nbin=30, text=""):
    """ Function to plot the interaction matrix"""
    #bins = np.linspace(np.ndarray.flatten(matx).min(), np.ndarray.flatten(matx).max(), nbin)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,7))
    figfunc.plotmat(matx, fig, ax, genes_order, text)
    #sns.histplot(np.ndarray.flatten(matx), ax=ax[1], stat="density", bins=bins)
    plt.show()
    
def to_thr_matrix(matrix, thr=0.02):
    """Function to set the interactions values below thr to 0
    """
    thr_matrix = np.copy(matrix)
    #Set to zero too small absolute values
    thr_matrix[np.abs(thr_matrix) <  (thr*np.nanmax(np.abs(matrix)))] = 0
    return thr_matrix
  
def interactions_plot(matx, matx_info, thr=0.01, nbins=30):
    thx_perc = thr*np.max(matx)
    # print(thx_perc)
    J_lin = matx.flatten()
    J_mean = np.mean(J_lin)
    J_std = np.std(J_lin)

    plt.ioff()  # turn off interactive mode
    n_J, bins_J, _ = plt.hist(J_lin, bins=nbins, 
                  density=True,  histtype= "stepfilled",
                 label="Couplings", color="navy")  # 
    plt.close() 

    centroids_J = (bins_J[1:] + bins_J[:-1]) / 2


    plt.figure(figsize=(8,7))
    plt.plot(centroids_J, n_J, color = "navy", lw = 5)

    plt.axvline(np.mean(J_lin), color='orangered', linestyle='-', lw = 3,
                label="$\mu$")#+str(np.round(J_mean,3))),
    plt.axvspan(J_mean-3*J_std, J_mean+3*J_std, 
                         alpha = 0.15, label="[$\mu-3\sigma$, $\mu+3\sigma$]",facecolor = "orangered", lw = 2)
    
    plt.axvline(thx_perc, color='grey', linestyle='-', lw=3, label = str(int(100*thr))+"\% of the max")
    plt.axvline(-thx_perc, color='grey', linestyle='-', lw=3)
    
    sns.scatterplot(x = matx_info[2,matx_info[3,:]==1],y=np.ones_like(1),   color="forestgreen", lw=3, 
                    label="true interactions", zorder=1, s=150)
    sns.scatterplot(x = matx_info[2,matx_info[3,:]==0],y=np.ones_like(1),   color="orangered", lw=3, 
                    label="false interactions", zorder=1, s=150)
    
    plt.grid()

    plt.xlabel('$J_{i,j}$', fontsize=20)
    plt.ylabel('$P(J_{i,j})$', fontsize=20)
    plt.tick_params(axis='both', labelsize=18)
    plt.legend(fontsize= 17, loc="upper left")

    
    
# ------ ADJACENCY MATRIX ------

def to_adj_matrix(matrix, thr=0.02):
    """Function to go from the interaction matrix to the adjacency matrix
    values with absolute value below the thr = 0
    negative values = -1
    positive values = +1
    """
    binary_matrix = np.copy(matrix)
    #Set to zero too small absolute values
    binary_matrix[np.abs(binary_matrix) <  (thr*np.max(np.abs(matrix)))] = 0
    # Set negative values to -1
    binary_matrix[binary_matrix <= -(thr*np.max(np.abs(matrix)))] = -1
    # Set positive values to 1
    binary_matrix[binary_matrix >=  (thr*np.max(np.abs(matrix)))] = 1
   
    return binary_matrix

def adj_plot(adj_matx):
    plt.figure(figsize=(9,7))
    plt.imshow(adj_matx, cmap = 'coolwarm')
    plt.xticks(np.linspace(0,23,24) , genes_order, rotation="vertical", fontsize=16)
    plt.yticks(np.linspace(0,23,24) , genes_order, rotation="horizontal", fontsize=16)
    plt.colorbar()
    plt.show()
    
# ------ RECIPROCITY -------
    
def reciprocity(adj_matrix):
    """ Function to compute the reciprocity of the Network
    INPUT:
    - adj_matrix: adjacency matrix
    
    OUTPUT:
    - num of total undirected edges/ num of total edges in the matrix"""
    # Compute the number of edges
    num_edges = np.count_nonzero(adj_matrix)
    # Compute the number of reciprocated edges WITHOUT considering the diagonal elements
    num_reciprocated = np.count_nonzero(adj_matrix * adj_matrix.T) - np.count_nonzero(adj_matrix.diagonal())
    # Compute and return the reciprocity
    return round(num_reciprocated / num_edges,2)

def node_reciprocity(adj_matrix, node_index):
    # Extract the subgraph of the network that contains only the interactions involving the specific node
    node_subgraph_row = adj_matrix[node_index, :]
    node_subgraph_col = adj_matrix[:, node_index]
    # Compute and return the reciprocity of the subgraph
    num_edges = np.count_nonzero(node_subgraph_row) + np.count_nonzero(node_subgraph_col)
    if num_edges == 0:
        return 0
    num_reciprocated = np.count_nonzero(node_subgraph_row * node_subgraph_col) - 1
    return round(num_reciprocated*2 / num_edges,2)


def nodes_reciprocity(matx, net_reciprocity, genes_list=genes_order):
    node_reciprocity_value = np.zeros(24)
    for ii in range(len(genes_list)):
        node_reciprocity_value[ii] = node_reciprocity(matx, ii)
        
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(y=net_reciprocity, xmin=0, xmax=1, color='lightblue', linestyle='-.', label='Network mean')
    ax.plot(genes_list, node_reciprocity_value, "o", markersize=8, markeredgecolor="darkblue", markerfacecolor="darkblue")
    ax.set_ylim([0,1])
    # get the current x-tick labels
    xtick_labels = ax.get_xticklabels()
    # set the x-tick labels
    ax.set_xticklabels(genes_list, rotation=45, fontsize=16)
    # Add grid to the plot
    ax.grid(True, linestyle='--')
    # Add a title and labels to the plot
    ax.set_title("Node Reciprocity Value", fontsize=20)
    ax.set_xlabel("Genes", fontsize=16)
    ax.set_ylabel("Node Reciprocity Value", fontsize=16)
    # Increase the margin of the plot
    plt.tight_layout()

    # Add a horizontal line at the mean value of the first 5 genes
    mean_Ngenes = np.mean(node_reciprocity_value[:8])
    std_Ngenes = np.std(node_reciprocity_value[:8])
    ax.axhline(y=mean_Ngenes, xmin=0, xmax=.33, color='b', linestyle='-', label='mean of Naive genes')
    
    mean_Fgenes = np.mean(node_reciprocity_value[8:14])
    std_Fgenes = np.std(node_reciprocity_value[8:14])
    ax.axhline(y=mean_Fgenes, xmin=0.31, xmax=.61, color='darkgreen', linestyle='-', label='mean of Formative genes')
    
    mean_Cgenes = np.mean(node_reciprocity_value[14:20])
    std_Cgenes = np.std(node_reciprocity_value[14:20])
    ax.axhline(y=mean_Cgenes, xmin=0.6, xmax=0.81, color='darkred', linestyle='-', label='mean of Committed genes')
    
    mean_Ogenes = np.mean(node_reciprocity_value[20:])
    std_Ogenes = np.std(node_reciprocity_value[20:])
    ax.axhline(y=mean_Ngenes, xmin=0.81, xmax=1, color='darkorange', linestyle='-', label='mean of Other genes')

    # Add legend
    ax.legend(loc='best', fontsize=16,bbox_to_anchor=(1.01, 1))

    ax.set_ylim([node_reciprocity_value.min()-0.2,node_reciprocity_value.max()+0.2])

    plt.show()
    return(node_reciprocity_value, [mean_Ngenes, mean_Fgenes, mean_Cgenes, mean_Ogenes], 
           [std_Ngenes, std_Fgenes, std_Cgenes, std_Ogenes] )

# ------ Degree -------------------

def plot_degree_distribution(matx):
    """
    Plot the degree distribution of a given matrix
    
    Parameters:
    matx: np.ndarray, The input matrix
    
    Returns:
    None
    """
    # Compute absolute value of the input matrix
    abs_matx = np.abs(matx)
    
    # Create subplots with 2 columns
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(7,12))
    
    #--- OUT ----------------------------
    bins_out = np.arange(np.floor(abs_matx.sum(axis=0).min())-0.5, np.ceil(abs_matx.sum(axis=0).max() + .5), 1)
    # Compute mean and standard deviation of Out-Degree distribution
    mean_out = abs_matx.sum(axis=0).mean()
    std_out = abs_matx.sum(axis=0).std()
    # Plot mean Out-Degree and mean plus/minus one standard deviation as vertical spans
    ax[0].axvline(mean_out, color="firebrick", 
                  label="mean \n OUT Degree\n ="+str(np.round(mean_out,2)), lw=3)
    ax[0].axvspan(mean_out - std_out, mean_out + std_out,  
                  alpha=0.15, facecolor = "orangered",
                  label="[$\mu-\sigma$, $\mu+\sigma$]")
    ax[0].hist(abs_matx.sum(axis=0), bins=bins_out, color="indianred", edgecolor='darkgray', linewidth=1.5, density=True)
    # Add legend and labels to Out-Degree histogram
    ax[0].legend(fontsize=16)
    ax[0].set_xlabel("Out degree", fontsize=18)
    ax[0].set_ylabel("pdf", fontsize=18)
    
    #--- IN ----------------------------
    bins_in = np.arange(np.floor(abs_matx.sum(axis=1).min())-0.5, np.ceil(abs_matx.sum(axis=1).max() + .5), 1)

    # Compute mean and standard deviation of In-Degree distribution
    mean_in = abs_matx.sum(axis=1).mean()
    std_in = abs_matx.sum(axis=1).std()
    # Plot mean In-Degree and mean plus/minus one standard deviation as vertical spans
    ax[1].axvline(mean_in, color="darkblue", 
                  label="mean \n IN Degree\n ="+str(np.round(mean_in,2)), lw=3)
    ax[1].axvspan(mean_in - std_in, mean_in + std_in, 
                  color='steelblue', alpha=0.15,
                 label="[$\mu-\sigma$, $\mu+\sigma$]")
    ax[1].hist(abs_matx.sum(axis=1), bins=bins_in, edgecolor='darkgray', linewidth=1.5, density=True)
    ax[1].legend(fontsize=16)
    ax[1].set_xlabel("In degree", fontsize=18)
    ax[1].set_ylabel("pdf", fontsize=18)

    plt.show()
    
    

# ------ DEGREE -------------------
def plot_indegree_outdegree(adj_matrix):
    adj_matrix = np.abs(adj_matrix)
    indegree = np.sum(adj_matrix, axis=1)
    outdegree = np.sum(adj_matrix, axis=0)
    print(indegree.max(), outdegree.max())
    plt.figure(figsize=(10,10))
    plt.scatter(outdegree, indegree, s=50)
    color_dict = {"darkblue": "Naive", "Darkred": "Formative", "darkgreen": "Committed", "darkorange": "Other"}
    color_list = []
    label_list = []
    for i, txt in enumerate(range(adj_matrix.shape[0])):
        if genes_order[txt] in naive:
            color = "darkblue"
        elif genes_order[txt] in formative:
            color = "Darkred"
        elif genes_order[txt] in committed:
            color = "darkgreen"
        else:
            color = "darkorange"
        plt.annotate(genes_order[txt], (outdegree[i]+0.25, indegree[i]+0.35), fontsize=15, color=color)
        color_list.append(color)
        label_list.append(genes_order[txt])
    legend_elements = [Line2D([0], [0], marker='o', color=c, label=color_dict[c], markersize=10) for c in np.unique(color_list)]
    plt.legend(handles=legend_elements, fontsize=18)
        
    plt.plot(np.linspace(3,24, 25), np.linspace(3,24, 25))
    
    plt.axvline(18, ls = '--', color = "dimgray", alpha = 0.4, lw = 5, label="18")
    plt.axhline(18, ls = '--', color = "dimgray", alpha = 0.4, lw = 5, label="18")
    
    plt.xlabel("OUT Degree", fontsize=18)
    plt.ylabel("IN Degree", fontsize=18)
    plt.grid(linestyle='--')
    plt.title("In-degree vs Out-degree", fontsize=20)
    plt.xticks(np.arange(min(indegree.min(), outdegree.min()), max(indegree.max(), outdegree.max())+1, 1))
    plt.yticks(np.arange(min(indegree.min(), outdegree.min()), max(indegree.max(), outdegree.max())+1, 1))
    plt.xlim([min(indegree.min(), outdegree.min())-1, max(indegree.max(), outdegree.max())+1])
    plt.ylim([min(indegree.min(), outdegree.min())-1, max(indegree.max(), outdegree.max())+1])
    plt.show()
    return(outdegree, indegree, spearmanr(outdegree, indegree)[0])


def distr_plot(ax, counts, bins, counts_std, text="average J", colorL="firebrick", colorSTD="orangered"):
    centroids = (bins[1:] + bins[:-1]) / 2

    ax.plot(centroids, counts, color = colorL, lw = 5, label =text )
    neg_std=np.zeros(len(counts_std))
    for i in range(len(counts_std)):
        neg_std[i] = max(0, counts[i]-counts_std[i])
    ax.fill_between(centroids, counts+counts_std, neg_std,
                     alpha = 0.3, color = colorSTD, lw = 2)
    ax.axvline(0, ls = '--', color = "dimgray", alpha = 0.7, lw = 5, label="zero")

    ax.legend(fontsize=20)
    ax.set_xlabel('$J_{i,j}$', fontsize=20)
    ax.set_ylabel('$P(J_{i,j})$', fontsize=20)


def plot_histograms(lN_high_meanMatx):
    plt.ioff() 
    OUT_counts, OUT_bins, _ = plt.hist(lN_high_meanMatx, bins=20, density=True)  
    plt.close()
    OUT_counts_mean  = np.mean(OUT_counts, axis=0)
    OUT_counts_std = np.std(OUT_counts, axis=0)

    plt.ioff() 
    IN_counts, IN_bins, _ = plt.hist(lN_high_meanMatx.T, bins=20,
                                    density=True)  
    plt.close()
    IN_counts_mean = np.mean(IN_counts,axis=0)
    IN_counts_std = np.std(IN_counts, axis = 0)
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
    
    distr_plot(ax1, OUT_counts_mean, OUT_bins, OUT_counts_std, text="average OUT\ninteraction stenght")
    distr_plot(ax2, IN_counts_mean, IN_bins, IN_counts_std, text="average IN\ninteraction stenght", colorL="royalblue", colorSTD="cornflowerblue")
    
    max_counts = max(np.max(IN_counts_mean), np.max(OUT_counts_mean))
    ax1.set_ylim([0, max_counts+1.2])
    ax2.set_ylim([0, max_counts+1.2])
    
    plt.show()
    

    
# ------ INTERACTIONS FORMAT ------

def string_list_to_tuple(string_list):
    """ Function to get a list of tuples from the standard format of the interactions
    INPUT:
    string list: list of strings made as follows 'geneFrom geneTo +/-1'
    +1 is for an activatory interaction
    -1 is for an inhibitory interaction
    
    OUTPUT: list of tuples made as follows (geneFrom, GeneTo)
    
    note: here the interaction sign is lost"""
    
    return [(x.split()[0], x.split()[1]) for x in string_list]


# ------ NETWORK VISUALIZATION -------

def visualize_graph_selNode(adj_matrix, node_names, naive_nodes, formative_nodes, committed_nodes, sel_node):
    """ Function to visualize all the network hilighting a specific selected node connections.
    - blue for inhibitory links;
    - red for excitatory links."""
    # Create a directed graph from the adjacency matrix
    adj_matrix = adj_matrix.T
    G = nx.DiGraph(adj_matrix)
    # Relabel the nodes with their new labels
    G = nx.relabel_nodes(G, {i:node_name for i, node_name in enumerate(node_names)})
    # Draw the graph using the circular layout
    plt.figure(figsize=(14,14))
    pos = nx.circular_layout(G)
    # Create a dictionary of node-to-color mappings
    color_map = {node: "lightblue" if node in naive_nodes else "lightgreen" if node in formative_nodes else "darkred" if node in committed_nodes else "darkorange" for node in G.nodes()}
    # Draw the nodes with different colors
    nx.draw_networkx_nodes(G, pos, node_color=color_map.values(), node_size=3000)
    nx.draw_networkx_labels(G, pos, labels={node:node for node in G.nodes()}, font_size=20)
    
    # Get the index of the selected node
    sel_node_idx = np.where(node_names==sel_node)[0][0]
    edges = [(u,v) for u,v in G.edges() if u == sel_node or v == sel_node]
    edge_colors = ['b' if adj_matrix[list(node_names).index(v), list(node_names).index(u)] == -1 else 'r' for u,v in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors,  arrowstyle='->', arrowsize=100, width=3)
    
    # Draw remaining edges
    edges = G.edges()
    sel_node_edges = [(u, sel_node) for u in G.predecessors(sel_node)] + [(sel_node, v) for v in G.successors(sel_node)]
    edges_to_remove = set(sel_node_edges)
    remaining_edges = [edge for edge in edges if edge not in edges_to_remove]
    nx.draw_networkx_edges(G, pos, edgelist=remaining_edges, arrowstyle='->', arrowsize=40, edge_color='lightgrey', alpha=0.7, width=1)
    plt.show()
    

def visualize_graphTrue(adj_matrix, node_names, naive_nodes, formative_nodes, committed_nodes, interactions=[], title=""):
    """ Fucntion to visualize the network of the known interaction trom a list"""
    # Create a directed graph from the adjacency matrix
    G = nx.DiGraph(adj_matrix.T)
    # Relabel the nodes with their new labels
    G = nx.relabel_nodes(G, {i:node_name for i, node_name in enumerate(node_names)})
    # Draw the graph using the circular layout
    plt.figure(figsize=(14,14))
    pos = nx.circular_layout(G)
    # Create a dictionary of node-to-color mappings
    color_map = {node: "lightskyblue" if node in naive_nodes else "palegoldenrod" if node in formative_nodes else "salmon" if node in committed_nodes else "silver" for node in G.nodes()}
    # Draw the nodes with different colors
    nx.draw_networkx_nodes(G, pos, node_color=color_map.values(), node_size=3500)
    nx.draw_networkx_labels(G, pos, labels={node:node for node in G.nodes()}, font_size=22)
    # Draw only the given interactions if they exist
    edges_to_draw_positive = []
    edges_to_draw_negative = []
    for interaction in interactions:
        node1, node2, direction = interaction.split(" ")
        if direction == "1":
            edges_to_draw_positive.append((node1, node2))
        else:
            edges_to_draw_negative.append((node1, node2))
    nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw_positive, arrowstyle='->', arrowsize=120, edge_color='r', alpha=0.7, width=3)
    nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw_negative, arrowstyle='->', arrowsize=120, edge_color='b', alpha=0.7, width=3)
    plt.title(title, fontsize=34)
    plt.show()
    


def visualize_graphSel(adj_matrix, node_names, naive_nodes, formative_nodes, committed_nodes, interactions=[], title=""):
    """ Function to visualize the network of the known interaction CORRETLY INFERRED from a list  
    note: here we plot only the nodes that have at least one connection in edges_to_draw_positive or edges_to_draw_negative"""
    # Create a directed graph from the adjacency matrix
    G = nx.DiGraph(adj_matrix.T)
    # Relabel the nodes with their new labels
    G = nx.relabel_nodes(G, {i:node_name for i, node_name in enumerate(node_names)})
    
    # Get the nodes to keep based on the given interactions
    nodes_to_keep = set()
    for interaction in interactions:
        node1, node2, direction = interaction.split(" ")
        nodes_to_keep.add(node1)
        nodes_to_keep.add(node2)
    # Remove the nodes that do not have at least one connection in edges_to_draw_positive or edges_to_draw_negative
    nodes_to_remove = set(G.nodes()) - nodes_to_keep
    G.remove_nodes_from(nodes_to_remove)
    
    # Draw the graph using the circular layout
    plt.figure(figsize=(14,14))
    pos = nx.circular_layout(G)
    
    # Create a dictionary of node-to-color mappings
    color_map = {node: "lightskyblue" if node in naive_nodes else "palegoldenrod" if node in formative_nodes else "salmon" if node in committed_nodes else "silver" for node in G.nodes()}
    
    # Draw the nodes with different colors
    nx.draw_networkx_nodes(G, pos, node_color=color_map.values(), node_size=3500)
    nx.draw_networkx_labels(G, pos, labels={node:node for node in G.nodes()}, font_size=22)
    
    # Draw only the given interactions if they exist
    edges_to_draw_positive = []
    edges_to_draw_negative = []
    for interaction in interactions:
        node1, node2, direction = interaction.split(" ")
        if direction == "1" and adj_matrix[list(node_names).index(node2), list(node_names).index(node1)]>0:
            edges_to_draw_positive.append((node1, node2))
        elif direction == "-1" and adj_matrix[list(node_names).index(node2), list(node_names).index(node1)]<0:
            edges_to_draw_negative.append((node1, node2))
    nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw_positive, arrowstyle='->', arrowsize=120, edge_color='r', alpha=0.7, width=3)
    nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw_negative, arrowstyle='->', arrowsize=120, edge_color='b', alpha=0.7, width=3)
    plt.title(title, fontsize=34)
    plt.show()

def visualize_graphSelTrue(adj_matrix, node_names, naive_nodes, formative_nodes, committed_nodes, interactions=[], title=""):
    """ Function to visualize the network of the known interaction from a list  
    note: here we plot only the nodes that have at least one connection in edges_to_draw_positive or edges_to_draw_negative"""
    # Create a directed graph from the adjacency matrix
    G = nx.DiGraph(adj_matrix.T)
    # Relabel the nodes with their new labels
    G = nx.relabel_nodes(G, {i:node_name for i, node_name in enumerate(node_names)})
    
    # Get the nodes to keep based on the given interactions
    nodes_to_keep = set()
    for interaction in interactions:
        node1, node2, direction = interaction.split(" ")
        nodes_to_keep.add(node1)
        nodes_to_keep.add(node2)
    # Remove the nodes that do not have at least one connection in edges_to_draw_positive or edges_to_draw_negative
    nodes_to_remove = set(G.nodes()) - nodes_to_keep
    G.remove_nodes_from(nodes_to_remove)
    
    # Draw the graph using the circular layout
    plt.figure(figsize=(14,14))
    pos = nx.circular_layout(G)
    
    # Create a dictionary of node-to-color mappings
    color_map = {node: "lightskyblue" if node in naive_nodes else "palegoldenrod" if node in formative_nodes else "salmon" if node in committed_nodes else "silver" for node in G.nodes()}
    
    # Draw the nodes with different colors
    nx.draw_networkx_nodes(G, pos, node_color=color_map.values(), node_size=3500)
    nx.draw_networkx_labels(G, pos, labels={node:node for node in G.nodes()}, font_size=22)
    
    # Draw only the given interactions if they exist
    edges_to_draw_positive = []
    edges_to_draw_negative = []
    for interaction in interactions:
        node1, node2, direction = interaction.split(" ")
        if direction == "1":
            edges_to_draw_positive.append((node2, node1))
        elif direction == "-1":
            edges_to_draw_negative.append((node1, node2))
    nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw_positive, arrowstyle='->', arrowsize=120, edge_color='r', alpha=0.7, width=3)
    nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw_negative, arrowstyle='->', arrowsize=120, edge_color='b', alpha=0.7, width=3)
    plt.title(title, fontsize=34)
    plt.show()

def string_list_to_tuple(string_list):
    """ to change the known interaction format:
    - from: "geneA geneB +/-1"
    - to: ("geneA", "geneB")"""
    return [(x.split()[0], x.split()[1]) for x in string_list]

def visualize_graph_KnownInferred(adj_matrix, node_names, naive_nodes, formative_nodes, committed_nodes, interactions=[], text="Inferred Known Interactions"):
    # Create a directed graph from the adjacency matrix
    adj_matrix = adj_matrix.T
    G = nx.DiGraph(adj_matrix)
    # Relabel the nodes with their new labels
    G = nx.relabel_nodes(G, {i:node_name for i, node_name in enumerate(node_names)})
    # Draw the graph using the circular layout
    plt.figure(figsize=(14,14))
    pos = nx.circular_layout(G)
    # Create a dictionary of node-to-color mappings
    color_map = {node: "lightblue" if node in naive_nodes else "lightgreen" if node in formative_nodes else "darkred" if node in committed_nodes else "darkorange" for node in G.nodes()}
    # Draw the nodes with different colors
    nx.draw_networkx_nodes(G, pos, node_color=color_map.values(), node_size=2000)
    nx.draw_networkx_labels(G, pos, labels={node:node for node in G.nodes()}, font_size=20)
    # Draw only the given interactions if they exist
    edges_to_draw = []
    for interaction in interactions:
        if interaction in G.edges():
            edges_to_draw.append(interaction)
    # Create a dictionary of edge-to-color mappings
    edge_color_map = {edge: "red" if adj_matrix[np.where(node_names==edge[0])[0][0], np.where(node_names==edge[1])[0][0]] > 0 else "blue" for edge in edges_to_draw}
    nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, arrowstyle='->', arrowsize=70, alpha=0.7, edge_color=edge_color_map.values())
    plt.title(text, fontsize=24)
    plt.show()

    
# ------ HUBS ------
def highest_OUTdegree_nodes(adj_matrix, n, genes):
    # Compute the out-degree of each node by summing the rows of the adjacency matrix
    outdegree = np.sum(np.abs(adj_matrix), axis=0)
    # Create a list of tuples where each tuple contains a node index and its out-degree
    outdegree_nodes = [[genes[i], outdegree[i], int([i][0])] for i in range(adj_matrix.shape[0])]
    # Sort the list of tuples by out-degree in descending order
    outdegree_nodes.sort(key=lambda x: x[1], reverse=True)
    # Return the first n nodes with the highest out-degree
    return np.array(outdegree_nodes[:n])

def highest_INdegree_nodes(adj_matrix, n, genes):
    # Compute the in-degree of each node by summing the rows of the adjacency matrix
    indegree = np.sum(np.abs(adj_matrix), axis=1)
    # Create a list of tuples where each tuple contains a node index and its in-degree
    indegree_nodes = [[genes[i], indegree[i], [i][0]] for i in range(adj_matrix.shape[0])]
    # Sort the list of tuples by in-degree in descending order
    indegree_nodes.sort(key=lambda x: x[1], reverse=True)
    # Return the first n nodes with the highest in-degree
    return np.array(indegree_nodes[:n])


