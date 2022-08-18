import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
import os 

import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils

import matplotlib.pyplot as plt
from cora_loader import Cora
from citeseer_loader import CiteSeer
from pubmed_loader import PubMed
from coauthor_cs_loader import CoauthorCS
from coauthor_physics_loader import  CoauthorPhysics
from amazon_photo_loader import AmazonPhoto
from amazon_computers_loader import AmazonComputers

from train import ModelTraining
from test import ModelEvaluation
from utils import UtilFunctions

from sklearn.manifold import TSNE
from model import GAT
import argparse

def argument_parser():

    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', help = 'enter name of dataset in smallcase', default = 'cora', type = str)
    parser.add_argument('--lr', help = 'learning rate', default = 0.2, type = float)
    parser.add_argument('--seed', help = 'Random seed', default = 100, type = int)
    parser.add_argument('--num_layers', help = 'number of hidden layers', default = 2, type = int)
    parser.add_argument('--hidden_dim', help = 'hidden dimension for node features', default = 16, type = int)
    parser.add_argument('--train_iter', help = 'number of training iteration', default = 100, type = int)
    parser.add_argument('--test_iter', help = 'number of test iterations', default = 1, type = int)
    parser.add_argument('--use_saved_model', help = 'use saved model in directory', default = False, type = None)
    parser.add_argument('--nheads', help = 'Number of attention heads', default = False, type = int)
    parser.add_argument('--alpha', help = 'slope of leaky relu', default = False, type = float)
    parser.add_argument('--dropout', help = 'Dropoout in the layers', default = 0.60, type = float)
    parser.add_argument('--w_decay', help = 'Weight decay for the optimizer', default = 0.0005, type = float)
    parser.add_argument('--device', help = 'cpu or gpu device to be used', default = 'cpu', type = None)

    return parser

parsed_args = argument_parser().parse_args()

dataset = parsed_args.dataset
lr = parsed_args.lr
seed = parsed_args.seed
num_layers = parsed_args.num_layers
hidden_dim = parsed_args.hidden_dim
train_iter = parsed_args.train_iter
test_iter = parsed_args.test_iter
use_saved_model = parsed_args.use_saved_model
nheads = parsed_args.nheads
alpha = parsed_args.alpha
dropout = parsed_args.dropout
weight_decay = parsed_args.w_decay
device = parsed_args.device
print("Device: ", device)


# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# if device == 'cuda:0':
#     torch.cuda.manual_seed(seed)

if dataset == 'cora':

    data_obj = Cora()

elif dataset == 'citeseer':

    data_obj = CiteSeer()

elif dataset == 'pubmed':

    data_obj = PubMed()

elif dataset == 'coauthorcs':

    data_obj = CoauthorCS()

elif dataset == 'coauthorphysics':

    data_obj = CoauthorPhysics()

elif dataset == 'amazonphoto':

    data_obj = AmazonPhoto()

elif dataset == 'amazoncomputers':

    data_obj = AmazonComputers()

else:

    print("Incorrect name of dataset")


# adjacency matrix generation
adj_matrix = UtilFunctions().adj_generation(data_obj.edge_index, data_obj.num_nodes, data_obj.num_edges)
# adj_matrix = data_obj.edge_index
model = GAT(data_obj.num_features, hidden_dim, data_obj.num_classes, num_layers, dropout, alpha, nheads, device)
model.to(device)
opti = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

print("Model Name: GAT++")
print("Dataset:", dataset.upper())
print("Hidden Layers:", num_layers)

if use_saved_model == 'False':

    # training of the model
    print("Optimization started....")
    trainer = ModelTraining()
    model_path = trainer.train(model, data_obj, adj_matrix, train_iter, opti, num_layers, device)

else:

    print("Trained model loaded from the directory...")
    model_path = os.getcwd() + "/saved_models/" + data_obj.name.lower() + "_" + str(num_layers) + "_layers_.pt"


# evaluation
print("Evaluating on Test set")
avg_acc = 0.0
max_acc = 0.0
eval = ModelEvaluation()

for i in range(test_iter):

    acc = eval.test(model, data_obj, adj_matrix, num_layers, model_path, device, is_validation = False)
    if acc > max_acc:
        max_acc = acc
    avg_acc += acc
    print("Test iteration:", i+1, " complete --- accuracy ",acc)

avg_acc /= test_iter

print(f'Maximum accuracy on Test set: {max_acc:.4f}')
print(f'Average accuracy on Test set: {avg_acc:.4f}')




