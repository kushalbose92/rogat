import ogb
import torch
# import ogb.nodeproppred
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader
from torch_geometric.loader import NeighborSampler, NeighborLoader, GraphSAINTSampler

class OGB_Loader():
    def __init__(self, d_name):
        
        dataset = PygNodePropPredDataset(name = d_name) 
        data = dataset[0]

        self.data = data
        self.name = d_name
        self.length = len(dataset)
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        self.num_nodes = data.num_nodes
        self.num_edges = data.num_edges
        # self.avg_node_degree = (data.num_edges / data.num_nodes)

        split_idx = dataset.get_idx_split()
        self.train_idx, self.valid_idx, self.test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        print(len(self.train_idx), " ", len(self.valid_idx), " ", len(self.test_idx))
        # self.train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
        # self.valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
        # self.test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)


        # self.train_mask = self.mask_generation(train_idx, self.num_nodes)
        # self.val_mask = self.mask_generation(val_idx, self.num_nodes)
        # self.test_mask = self.mask_generation(test_idx, self.num_nodes)

        self.node_features = data.x
        self.node_labels = data.y
        # self.node_labels = self.node_labels.squeeze(1)
        self.edge_index = data.edge_index

    def mask_generation(self, index, num_nodes):
        mask = torch.zeros(num_nodes, dtype = torch.bool)
        mask[index] = 1
        return mask

    def edge_id_map(self, edge_index):

        num_edges = edge_index[0].shape[0]
        edge_dict = {}

        for e_idx in range(num_edges):
            s = edge_index[0][e_idx].item()
            t = edge_index[1][e_idx].item()
            edge_dict[(s, t)] = e_idx
    
        return edge_dict


'''
number of isolated nodes is = 17440 ---- ogbn-arxiv
'''
ogb = OGB_Loader('ogbn-products')
# edge_index = ogb.edge_index
# print(edge_index.shape)
# node_list = [0 for i in range(ogb.num_nodes)]
# for i in range(ogb.num_edges):
#     node_list[int(ogb.edge_index[0][i])] += 1
# print(node_list.count(0))
# print(sum(node_list))
# print("number of node ", ogb.num_nodes)
# print("number of edges ", ogb.num_edges)
# print("number of classes ", ogb.num_classes)
# print("number of features ", ogb.num_features)


# loader = GraphSAINTSampler(ogb.data, batch_size=6000,
#                                      num_steps=5, 
#                                      sample_coverage=100,
#                                      save_dir=None,
#                                      num_workers=4)
# c = 0
# for data in loader:
#     print(data)
#     c+=1
# print("total count ", c)

# train_loader = NeighborSampler(ogb.edge_index, 
#                                 node_idx=ogb.train_idx,
#                                 sizes=[15, 10, 5], 
#                                 batch_size=1024,
#                                 shuffle=True, 
#                                 num_workers=12)

# train_loader = NeighborSampler(ogb.edge_index, node_idx=None, sizes=[-1],
                                #   batch_size=4096, shuffle=False,
                                #   num_workers=12)


# print(train_loader)
# c = 0
# for (batch_size, nid, adj) in train_loader:
    # print(batch_size)
    # print(nid, "  ", nid.shape)
    # edge_index, eid, size = adj 
    # print(edge_index)
    # print(eid)
    # print(size)
    # print()
    # c += 1
    # print(c)

   

# print(ogb.node_features)
# print(ogb.train_loader)
# print(ogb.valid_loader)
# print(ogb.test_loader)
# print(ogb.train_mask.sum(), " ", ogb.val_mask.sum(), " ", ogb.test_mask.sum())


# for e in range(ogb.num_edges):
#     if ogb.edge_index[0][e] == torch.tensor(91710):
#         print(ogb.edge_index[1][e])