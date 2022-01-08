from torch_geometric import data
from torch_geometric.utils import dense_to_sparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

def to_torch_geom(adj, features, node_labels, graph_labels, device, debug):
    graphs = {}
    for key in adj.keys():      # train, val, test
        graphs[key] = []
        for i in range(len(adj[key])):          # Graph of a given size
            batch_i = []
            for j in range(adj[key][i].shape[0]):       # Number of graphs
                graph_adj = adj[key][i][j]
                graph = data.Data(x=features[key][i][j],
                                  edge_index=dense_to_sparse(graph_adj)[0],
                                  y=graph_labels[key][i][j].unsqueeze(0),
                                  pos=node_labels[key][i][j])
                if not debug:
                    batch_i.append(graph)
            if debug:
                batch_i.append(graph)
            graphs[key].append(batch_i)
    return graphs
