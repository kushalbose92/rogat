import argparse
import ogb

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from gat_source_code import GATConv
# from torch_geometric.nn import SuperGATConv
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, nheads):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, int(hidden_channels/nheads), nheads))

        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(int(hidden_channels/nheads), int(hidden_channels/nheads), nheads))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(GATConv(hidden_channels, out_channels, 1))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        embedding = x
        return x.log_softmax(dim=-1), embedding


def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out, embedding = model(data.x, data.adj_t)
    out = out[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out, embedding = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def visualize(feat_map, color, name, hidden_layers):
        z = TSNE(n_components=2).fit_transform(feat_map.detach().cpu().numpy())

        plt.figure(figsize=(10,10))
        plt.xticks([])
        plt.yticks([])

        plt.scatter(z[:, 0], z[:, 1], s=70, c=color.cpu(), cmap="Set2")
        plt.savefig(os.getcwd() + "/visuals/" + name + "_" + str(hidden_layers) + "_layers_embedding.png")
        plt.clf()
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--nheads', type = int, default = 2)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    data_name = 'ogbn-arxiv'

    dataset = PygNodePropPredDataset(name= data_name, transform = T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    
    # data.adj_t = data.edge_index
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    model_path = os.getcwd() + "/saved_models/" + data_name.lower() + "_" + str(args.num_layers) + "_layers_.pt"

    model = GAT(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout, args.nheads).to(device)

    # for name, param in model.named_parameters():
    #     print(name)
    #     print(param.shape)

    evaluator = Evaluator(name = data_name)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
        best_acc = 0.0
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)
            train_acc, valid_acc, test_acc = result
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), model_path)

            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

    # model loading and testing
    model.load_state_dict(torch.load(model_path))
    train_acc, valid_acc, test_acc = test(model, data, split_idx, evaluator)
    out, emb = model(data.x, data.adj_t)
    # visualize(emb, data.y, data_name, args.num_layers)
    print(f'Test accuracy: {100 * test_acc:.2f}')

if __name__ == "__main__":
    main()