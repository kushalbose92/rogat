import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# proposed architecture --- GAT++
class GraphAttentionLayer(nn.Module):
    
    def __init__(self, in_features, out_features, dropout, alpha, concat):
        super(GraphAttentionLayer, self).__init__()

        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        print("GAT++ layer")

        # weights for source node
        self.W_s = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W_s.data, gain=1.414)

        # weights for neighboring nodes
        self.W_n = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W_n.data, gain=1.414)

        # weights for estimate attention mechanism
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):

        Wh_s = torch.mm(h, self.W_s) 
        Wh_n = torch.mm(h, self.W_n) 
        
        e = self._prepare_attentional_mechanism_input(Wh_s, Wh_n)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh_n)

        if self.concat:
            return F.elu(h_prime), attention
        else:
            return h_prime, attention

    def _prepare_attentional_mechanism_input(self, Wh_s, Wh_n):
       
        Wh1 = torch.matmul(Wh_s, self.a[:self.out_features, :])   
        Wh2 = torch.matmul(Wh_n, self.a[self.out_features:, :])

        # broadcast add
        e = Wh1 + Wh2.T
        e = self.leakyrelu(e)
        return e

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


