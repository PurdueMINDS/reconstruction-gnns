import torch, time, sys
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Embedding, Sequential, ReLU, Linear, Dropout
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, PNAConv, BatchNorm
from torch_geometric.data import Batch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool, MessagePassing, PNAConv
from torch_geometric.utils import degree

class GCNConv(MessagePassing):
    def __init__(self, emb_dim, dim1, dim2):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(dim1, dim2)
        self.root_emb = torch.nn.Embedding(1, dim2)
        self.bond_encoder = Sequential(Linear(emb_dim, dim2), ReLU(), Linear(dim2, dim2))

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

class GINConv(MessagePassing):
    def __init__(self, emb_dim, dim1, dim2):
        super(GINConv, self).__init__(aggr="add")
        self.bond_encoder = Sequential(Linear(emb_dim, dim1), ReLU(), Linear(dim1, dim1))
        self.mlp = Sequential(Linear(dim1, dim1), ReLU(), Linear(dim1, dim2))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

class NetGCN(torch.nn.Module):
    def __init__(self, node_size, edge_size, hidden_size, out_size, num_layers):
        super(NetGCN, self).__init__()

        self.convs = ModuleList([GCNConv(edge_size, node_size, hidden_size)])
        self.batch_norms = ModuleList([BatchNorm(hidden_size)])
        for _ in range(num_layers-1):
            conv = GINConv(edge_size, hidden_size, hidden_size)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_size))

        self.fc1 = Linear(num_layers * hidden_size, hidden_size)
        self.fc2 = Linear(hidden_size, hidden_size)
        self.fc3 = Linear(hidden_size, hidden_size)
        self.fc4 = Linear(hidden_size, out_size)

    def forward(self, x, edge_index, edge_attr, batch):

        x_lst = []

        for conv, bn in zip(self.convs, self.batch_norms):
            x = F.relu(conv(x, edge_index, edge_attr))
            x = bn(x)
            x_lst.append(x)

        x = torch.cat(x_lst, dim=-1)
        x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc4(x)

class NetGINE(torch.nn.Module):
    def __init__(self, node_size, edge_size, hidden_size, out_size, num_layers):
        super(NetGINE, self).__init__()

        self.convs = ModuleList([GINConv(edge_size, node_size, hidden_size)])
        self.batch_norms = ModuleList([BatchNorm(hidden_size)])
        for _ in range(num_layers-1):
            conv = GINConv(edge_size, hidden_size, hidden_size)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_size))

        self.fc1 = Linear(num_layers * hidden_size, hidden_size)
        self.fc2 = Linear(hidden_size, hidden_size)
        self.fc3 = Linear(hidden_size, hidden_size)
        self.fc4 = Linear(hidden_size, out_size)

    def forward(self, x, edge_index, edge_attr, batch):

        x_lst = []

        for conv, bn in zip(self.convs, self.batch_norms):
            x = F.relu(conv(x, edge_index, edge_attr))
            x = bn(x)
            x_lst.append(x)

        x = torch.cat(x_lst, dim=-1)
        x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc4(x)

class NetSubgraphGCN(torch.nn.Module):
    def __init__(self, node_size, edge_size, hidden_size, out_size, num_layers):
        super(NetSubgraphGCN, self).__init__()

        self.non_linearity = nn.ReLU()

        self.convs = ModuleList([GCNConv(edge_size, node_size, hidden_size)])
        self.batch_norms = ModuleList([BatchNorm(hidden_size)])
        for _ in range(num_layers-1):
            conv = GINConv(edge_size, hidden_size, hidden_size)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_size))

        self.fc0 = Linear(hidden_size*num_layers, hidden_size)
        self.fc1 = Linear(hidden_size, hidden_size)
        self.fc2 = Linear(hidden_size, hidden_size)
        self.fc3 = Linear(hidden_size, hidden_size)
        self.fc4 = Linear(hidden_size, hidden_size)
        self.fc5 = Linear(hidden_size, hidden_size)
        self.fc6 = Linear(hidden_size, hidden_size)
        self.fc7 = Linear(num_layers*hidden_size, hidden_size)

        self.pred = Linear(hidden_size, out_size)

    def forward(self, x, edge_index, edge_attr, batch, weights, subgraph_batch):

        x_lst = []

        for conv, bn in zip(self.convs, self.batch_norms):
            x = F.relu(conv(x, edge_index, edge_attr))
            x = bn(x)
            x_lst.append(x)

        x = torch.cat(x_lst, dim=-1)

        x = global_mean_pool(x, batch)

        x = self.non_linearity(self.fc0(x))
        #x = self.non_linearity(self.fc1(x))
        #x = self.non_linearity(self.fc2(x))
        #x = self.non_linearity(self.fc3(x))

        x = x*weights
        x = global_add_pool( x, subgraph_batch )
        norm = global_add_pool( weights, subgraph_batch )
        x = x/norm # use mean

        x = self.non_linearity(self.fc4(x))
        x = self.non_linearity(self.fc5(x))
        #x = self.non_linearity(self.fc6(x))
        #x = self.non_linearity(self.fc7(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.pred(x)

class NetSubgraphGINE(torch.nn.Module):
    def __init__(self, node_size, edge_size, hidden_size, out_size, num_layers):
        super(NetSubgraphGINE, self).__init__()

        self.non_linearity = nn.ReLU()

        self.convs = ModuleList([GINConv(edge_size, node_size, hidden_size)])
        self.batch_norms = ModuleList([BatchNorm(hidden_size)])
        for _ in range(num_layers-1):
            conv = GINConv(edge_size, hidden_size, hidden_size)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_size))

        self.fc0 = Linear(hidden_size*num_layers, hidden_size)
        self.fc1 = Linear(hidden_size, hidden_size)
        self.fc2 = Linear(hidden_size, hidden_size)
        self.fc3 = Linear(hidden_size, hidden_size)
        self.fc4 = Linear(hidden_size, hidden_size)
        self.fc5 = Linear(hidden_size, hidden_size)
        self.fc6 = Linear(hidden_size, hidden_size)
        self.fc7 = Linear(num_layers*hidden_size, hidden_size)

        self.pred = Linear(hidden_size, out_size)

    def forward(self, x, edge_index, edge_attr, batch, weights, subgraph_batch):

        x_lst = []

        for conv, bn in zip(self.convs, self.batch_norms):
            x = F.relu(conv(x, edge_index, edge_attr))
            x = bn(x)
            x_lst.append(x)

        x = torch.cat(x_lst, dim=-1)

        x = global_mean_pool(x, batch)

        x = self.non_linearity(self.fc0(x))
        #x = self.non_linearity(self.fc1(x))
        #x = self.non_linearity(self.fc2(x))
        #x = self.non_linearity(self.fc3(x))

        x = x*weights
        x = global_add_pool( x, subgraph_batch )
        norm = global_add_pool( weights, subgraph_batch )
        x = x/norm # use mean

        x = self.non_linearity(self.fc4(x))
        x = self.non_linearity(self.fc5(x))
        #x = self.non_linearity(self.fc6(x))
        #x = self.non_linearity(self.fc7(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.pred(x)

class NetPNA(torch.nn.Module):
    def __init__(self, node_size, edge_size, out_size, deg, num_layers, dropout=True):
        super(NetPNA, self).__init__()

        hidden_size = 64

        self.node_emb = Embedding(node_size, hidden_size)
        self.edge_emb = Embedding(edge_size, hidden_size)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(num_layers):
            conv = PNAConv(in_channels=hidden_size, out_channels=hidden_size,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=hidden_size, towers=4, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_size))

        self.mlp = Sequential(Linear(hidden_size, hidden_size//2), ReLU(), Linear(hidden_size//2, hidden_size//4), ReLU(), Linear(hidden_size//4, out_size))
        if dropout: Sequential(Linear(hidden_size, hidden_size//2), ReLU(), Linear(hidden_size//2, hidden_size//4), ReLU(), Dropout(0.5), Linear(hidden_size//4, out_size))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch)
        return self.mlp(x)

class NetSubgraphPNA(torch.nn.Module):
    def __init__(self, node_size, edge_size, out_size, deg, k):
        super(NetSubgraphPNA, self).__init__()

        hidden_size = 16
        self.non_linearity = nn.ReLU()

        self.gnn = NetPNA(node_size, edge_size, 16, deg, k, dropout=False)

        self.fc1 = Linear(hidden_size, hidden_size)
        self.fc2 = Linear(hidden_size, hidden_size)
        self.pred = Linear(hidden_size, out_size)

    def forward(self, x, edge_index, edge_attr, batch, weights, subgraph_batch):

        x = self.gnn( x, edge_index, edge_attr, batch )

        x = x*weights
        x = global_add_pool( x, subgraph_batch )
        norm = global_add_pool( weights, subgraph_batch )
        x = x/norm #use mean
        x = self.non_linearity(self.fc1(x))
        x = self.non_linearity(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.pred(x)
