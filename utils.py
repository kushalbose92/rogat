import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils

import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.ticker as mtick
import matplotlib.cm as cm

from sklearn.manifold import TSNE
import math
import numpy as np
import torch
import os
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


class UtilFunctions():

    def __init__(self):

        return

    # visuals of embedding
    def visualize(self, feat_map, color, name, hidden_layers):
        z = TSNE(n_components=2).fit_transform(feat_map.detach().cpu().numpy())

        plt.figure(figsize=(10,10))
        plt.xticks([])
        plt.yticks([])

        plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
        plt.savefig(os.getcwd() + "/visuals/" + name + "_" + str(hidden_layers) + "_layers_embedding.png")
        plt.clf()
        plt.close()

    # loss function
    def loss_fn(pred, label):

        # return F.nll_loss(pred, label)
        return F.cross_entropy(pred, label)

    def adj_list_generation(self, edge_index, num_nodes, num_edges):

        adj_list = [[] for n in range(num_nodes)]

        for n in range(num_edges):

            src = int(edge_index[0][n])
            neigh = int(edge_index[1][n])
            adj_list[src].append(neigh)

        return adj_list

    def adj_generation(self, edge_index, num_nodes, num_edges):

        adj = torch.zeros(num_nodes, num_nodes)
        for e in range(num_edges):
            src = edge_index[0][e]
            tgt = edge_index[1][e]
            adj[src][tgt] = 1

        return adj + torch.eye(num_nodes)

    # plots histrogram of attention values 
    def attn_dist(self, attn_val, data_obj):

        attn_list = attn_val.reshape(-1, attn_val.shape[0]*attn_val.shape[1])
        attn_list = attn_list.detach().cpu().numpy()
        attn_list = attn_list[attn_list != 0.0]
        fileName = 'attn_hist_gat.txt'
        file = open(fileName, 'w')
        for i in range(len(attn_list)):
            values = str(attn_list[i]) + "\n"
            file.write(values)
        file.close()

    # compares attention values between GAT and GAT++
    def attn_comp(self, data_obj, attn_val):

        edge_index = data_obj.edge_index
        labels = data_obj.node_labels
        flag_list = []
        fileName = 'attn_val_gat.txt'
        for e in range(data_obj.num_edges):
            src = edge_index[0][e]
            tgt = edge_index[1][e]
            if labels[src] == labels[tgt]:
                flag_list.append(1)
            else:
                flag_list.append(0)
        one_count = flag_list.count(1)
        zero_count = flag_list.count(0)
        attn_val *= (torch.ones(attn_val.shape[0], attn_val.shape[0]).to(device) - torch.eye(attn_val.shape[0]).to(device))
        attn_list = attn_val.reshape(-1, attn_val.shape[0]*attn_val.shape[1])
        attn_list = attn_list.detach().cpu().numpy()
        attn_list = attn_list[attn_list != 0.0]
        colors = cm.rainbow(np.linspace(0, 1, 2))
        col_list = [colors[0] if flag_list[i] == 1 else colors[1] for i in range(len(flag_list))]
        flag_list = np.array(flag_list)
        file = open(fileName,'w')
        for i in range(len(attn_list)):
            values = str(flag_list[i]) + "," + str(attn_list[i]) + "\n"
            file.write(values)
        file.close()

