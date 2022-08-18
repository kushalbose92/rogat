import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer
from utils import UtilFunctions

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_layers, dropout, alpha, nheads, device):
    
        super(GAT, self).__init__()
        self.dropout = dropout
        self.device = device
        self.attn_layers = nn.ModuleList()
        self.num_layers = num_layers

        for l in range(self.num_layers):
            if l == 0:
                self.attn_layers.append(GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=False).to(self.device))
            elif l == self.num_layers - 1:
                self.attn_layers.append(GraphAttentionLayer(nhid, nclass, dropout=dropout, alpha=alpha, concat=False).to(self.device))
            else:
                self.attn_layers.append(GraphAttentionLayer(nhid, nhid, dropout=dropout, alpha=alpha, concat=False).to(self.device))

    def forward(self, x, adj, node_labels):

        x = x.to(self.device)
        adj = adj.to(self.device)
        x = F.dropout(x, self.dropout, training = self.training)

        for l in range(self.num_layers-1):
            x, attn_val = self.attn_layers[l](x, adj)
            # UtilFunctions.visualize(self, x, node_labels, 'cora', l)
            if l != self.num_layers - 1:
                x = F.elu(x)
            else:
                x = F.relu(x)

        # x = F.elu(out_x)
        x = F.dropout(x, self.dropout, training = self.training)
        x, attn_val = self.attn_layers[self.num_layers-1](x, adj)
        embedding = x
        x = F.log_softmax(x, dim = 1)
        return embedding, x, attn_val



# with attention heads
# class GAT(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, num_layers, dropout, alpha, nheads, device):
    
#         super(GAT, self).__init__()
#         self.dropout = dropout
#         self.device = device
#         self.attentions = nn.ModuleList()

#         for _ in range(nheads):
#             self.attentions.append(GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=False).to(self.device))

#         self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False).to(self.device)

#     def forward(self, x, adj, _):

#         x = x.to(self.device)
#         adj = adj.to(self.device)
#         x = F.dropout(x, self.dropout, training = self.training)

#         out_x = None
#         for head in range(len(self.attentions)):
    
#             attn_x, attn_val = self.attentions[head](x, adj)
            
#             if out_x == None:
#                 out_x = attn_x
#             else:
#                 out_x = torch.cat([out_x, attn_x], dim = 1)

#         x = F.elu(out_x)
#         x = F.dropout(x, self.dropout, training = self.training)
#         x, attn_val = self.out_att(x, adj)
#         embedding = x
#         # x = F.elu(x)
#         x = F.log_softmax(x, dim = 1)
#         return embedding, x, attn_val
