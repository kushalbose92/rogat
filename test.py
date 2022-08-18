import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np

import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils

import matplotlib.pyplot as plt
from cora_loader import Cora
from citeseer_loader import CiteSeer
from pubmed_loader import PubMed
# from coauthor_cs_loader import CoauthorCS
from sklearn.manifold import TSNE
import math
from utils import UtilFunctions


class ModelEvaluation():

    def __init__(self):

        return

    def test(self, model, data_obj, adj, num_layers, model_path, device, is_validation):

        if is_validation is False:
            # for test 
            model.load_state_dict(torch.load(model_path))
            mask = data_obj.test_mask
        else:
            # for validation 
            mask = data_obj.test_mask

        model.eval()
        correct = 0
        emb, pred, attn_val = model(data_obj.node_features, adj, data_obj.node_labels)
        pred = pred.argmax(dim = 1)
        label = data_obj.node_labels
        pred = pred[mask]
        label = label[mask]
        pred = pred.to(device)
        label = label.to(device)
        correct = pred.eq(label).sum().item()
        accuracy = correct / int(mask.sum())
        
        if is_validation is False:

            # UtilFunctions().visualize(emb, data_obj.node_labels, data_obj.name, num_layers)
            # UtilFunctions().attn_dist(attn_val, data_obj)
            UtilFunctions().attn_comp(data_obj, attn_val)

        return accuracy
