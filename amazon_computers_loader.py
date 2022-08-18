import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torch_geometric.datasets import Amazon
from torch_geometric.transforms import NormalizeFeatures

import torch
import torch.nn.functional as F

import random

class AmazonComputers():

    def __init__(self):

        dataset = Amazon(root='data/amazon_computers/', name = 'Computers', transform = NormalizeFeatures())
        data = dataset[0]

        self.data = data
        self.name = 'AmazonComputers'
        self.length = len(dataset)
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        self.num_nodes = data.num_nodes
        self.num_edges = data.num_edges
        self.avg_node_degree = (data.num_edges / data.num_nodes)
    
        self.contains_isolated_nodes = data.has_isolated_nodes()
        self.data_contains_self_loops = data.has_self_loops()
        self.is_undirected = data.is_undirected()

        self.node_features = data.x
        self.node_labels = data.y
        self.edge_index = data.edge_index

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


    def data_visualize(self, img_name):

        z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())

        plt.figure(figsize=(10,10))
        plt.xticks([])
        plt.yticks([])

        plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
        plt.savefig(img_name + ".png")
        plt.clf()

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
            train_set = sampled_c[:20]
            all_indices += sampled_c[20:]
            train_idx += train_set
    
        random.shuffle(all_indices)
        val_idx = all_indices[:300]
        test_idx = all_indices[300:]

        return train_idx, val_idx, test_idx

    def mask_generation(self, index, num_nodes):
        mask = torch.zeros(num_nodes, dtype = torch.bool)
        mask[index] = 1
        return mask


# data_obj = AmazonComputers()
# print("number of features = ", data_obj.num_features)
# print("number of nodes = ", data_obj.num_nodes)
# print("number of edges = ", data_obj.num_edges)
# print("number of classes = ", data_obj.num_classes)
# print(data_obj.train_mask.sum(), "  ", data_obj.val_mask.sum(), "  ", data_obj.test_mask.sum())
# print(data_obj.node_labels)
# print(data_obj.train_mask.shape)
# print(data_obj.val_mask.shape)
# print(data_obj.test_mask.shape)
# print(data_obj.edge_index)
# print(cora.train_mask.sum(), " ", cora.val_mask.sum(), " ", cora.test_mask.sum())
# print(cora.data)
# print(cora.edge_index[0][:30])
# print(cora.edge_index[1][:30])
# adj_list = cora.adj_list_generation(cora.edge_index)
# print(adj_list)
# path_list = cora.path_generator(adj_list, 64, 4)


# for n in range(cora.num_nodes):

    # print(n, "    ", adj_list[n])

# print("---------------------")

# for n in range(path_list.shape[0]):

#     print(path_list[n])

# print(path_list.shape)

# for e in range(cora.num_edges):

#     print(cora.edge_index[0][e], "   ", cora.edge_index[1][e])

# nodes = cora.num_nodes
# edges = cora.num_edges
# node_features = cora.node_features

# adj = torch.zeros(nodes, nodes)

# for e in range(edges):

    # src = cora.edge_index[0][e]
    # tgt = cora.edge_index[1][e]

    # print(cora.edge_index[0][e], "   ", cora.edge_index[1][e])

