import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torch_geometric.datasets import Coauthor
from torch_geometric.transforms import NormalizeFeatures

import torch
import torch.nn.functional as F

import random

class CoauthorCS():

    def __init__(self):

        dataset = Coauthor(root='data/coauthor_cs', name='CS', transform = NormalizeFeatures())
        data = dataset[0]

        self.data = data
        self.name = "CoauthorCS"
        self.length = len(dataset)
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        self.num_nodes = data.num_nodes
        self.num_edges = data.num_edges
        self.avg_node_degree = (data.num_edges / data.num_nodes)
        # self.train_label_rate = (int(self.train_mask.sum()) / data.num_nodes)
        self.contains_isolated_nodes = data.has_isolated_nodes()
        self.data_contains_self_loops = data.has_self_loops()
        self.is_undirected = data.is_undirected()

        self.node_features = data.x
        self.node_labels = data.y
        self.edge_index = data.edge_index
        # self.train_mask = data.train_mask

        train_idx, val_idx, test_idx = self.index_generation()
        self.train_mask = self.mask_generation(train_idx, self.num_nodes)
        self.val_mask = self.mask_generation(val_idx, self.num_nodes)
        self.test_mask = self.mask_generation(test_idx, self.num_nodes)

    # adjacency list generation
    def adj_list_generation(self, edge_index):

        adj_list = [[] for n in range(self.num_nodes)]
        src_list = edge_index[0]
        dest_list = edge_index[1]

        for n in range(self.num_edges):

            adj_list[int(src_list[n])].append(int(dest_list[n]))

        return adj_list

    def index_generation(self):

        class_idx = [[] for i in range(self.num_classes)]
        train_idx = []
        val_idx = []
        test_idx = []
        for n in range(self.num_nodes):
            
            class_idx[self.node_labels[n]].append(n)

        # z = [len(class_idx[i]) for i in range(len(class_idx))]
        # print(z)
        all_indices = []
        for c in range(self.num_classes):

            sampled_c = random.sample(class_idx[c], len(class_idx[c])) 
            random.shuffle(sampled_c)
            train_set = sampled_c[:40]
            all_indices += sampled_c[40:]
            train_idx += train_set
    
        random.shuffle(all_indices)
        val_idx = all_indices[:2250]
        test_idx = all_indices[2250:]

        return train_idx, val_idx, test_idx

    def mask_generation(self, index, num_nodes):
        mask = torch.zeros(num_nodes, dtype = torch.bool)
        mask[index] = 1
        return mask
    

# coauthor_cs = CoauthorCS()
# print("number of nodes ", coauthor_cs.num_nodes)
# print("number of edges ", coauthor_cs.num_edges)
# print("number of features ", coauthor_cs.num_features)
# print("number of classes ", coauthor_cs.num_classes)

# print(coauthor_cs.train_mask.sum(), "  ", coauthor_cs.val_mask.sum(), "   ", coauthor_cs.test_mask.sum())