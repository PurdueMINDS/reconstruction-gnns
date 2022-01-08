import torch, copy, random, itertools, time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from layers.gnn import *
from datasets_generation.build_cycles import FourCyclesDataset
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from random import sample
import pickle

rootdir = './data/datasets_kcycle_nsamples=10000/'
import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

""" Interactive Shell Command: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=urca-gpu --time=12:00:00 --pty bash """

""" Run: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=urca-gpu --output=deck1gcn.txt python -u deck-n-l-gcn.py 1 & """
""" Run: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=ipanema-gpu --output=deck2gcn.txt python -u deck-n-l-gcn.py 2 & """
""" Run: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=ml01-gpu,ml02-gpu,ml03-gpu,ml04-gpu,ml05-gpu,ml06-gpu,ml07-gpu,ml08-gpu --output=deck3gcn.txt python -u deck-n-l-gcn.py 3 & """
""" Run: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=ml01-gpu,ml02-gpu,ml03-gpu,ml04-gpu,ml05-gpu,ml06-gpu,ml07-gpu,ml08-gpu --output=deck-half-gcn.txt python -u deck-n-l-gcn.py half & """
#
# nohup python -u deck-n-l-gcn.py 3 4 36 > deck3-4gcn.txt &
# nohup python -u deck-n-l-gcn.py half 4 36 > deckhalf-4gcn.txt &

# nohup python -u deck-n-l-gcn.py 1 6 56 > deck1-6gcn.txt &
# nohup python -u deck-n-l-gcn.py 2 6 56 > deck2-6gcn.txt &
# nohup python -u deck-n-l-gcn.py 3 6 56 > deck3-6gcn.txt &
# nohup python -u deck-n-l-gcn.py half 6 56 > deckhalf-6gcn.txt &

""" Run: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=copa-gpu --output=deck1-8gcn.txt python -u deck-n-l-gcn.py 1 8 72 & """
""" Run: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=copa-gpu --output=deck2-8gcn.txt python -u deck-n-l-gcn.py 2 8 72 & """
""" Run: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=copa-gpu --output=deck3-8gcn.txt python -u deck-n-l-gcn.py 3 8 72 & """
""" Run: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=urca-gpu --output=deckhalf-8gcn.txt python -u deck-n-l-gcn.py half 8 72 & """

import math
ell = -1
if sys.argv[1].isdigit(): ell = int(sys.argv[1])

k = int(sys.argv[2])
graph_size = int(sys.argv[3])

train_dataset = list(FourCyclesDataset(k, graph_size, rootdir, proportion=1.0, train=True, transform=None))
test_dataset = list(FourCyclesDataset(k, graph_size, rootdir, proportion=1.0, train=False, transform=None))

device = torch.device("cuda:0")

node_size = 1
edge_size = 1
hidden_size = 300
batch_size = 16

train_sample_size = 10
test_sample_size = 10

results = []

def sample_combinations(choices, size, count, weights):
    collected = {tuple(sample(choices, size)) for _ in range(count)}
    while len(collected) < count:
        tup_sampled = tuple(random.choices(population=choices, k=size, weights=weights))
        while len(set(tup_sampled)) != len(tup_sampled):
            tup_sampled = tuple(random.choices(population=choices, k=size, weights=weights))
        collected.add(tup_sampled)
    return list(collected)

def get_subgraphs( dataset, sample_size):
    subgraphs = []
    for graph in dataset:
        global ell
        if ell == -1: ell = math.ceil(graph.num_nodes/2)
        if graph.x.size(0) > ell:
            weights = [1]*graph.num_nodes
            graph_subgraphs = []
            to_remove = sample_combinations(range(graph.num_nodes), ell, min([ sample_size, ncr(graph.num_nodes,ell)  ]), weights)
            for l in to_remove:
                deck_nodes = list(range(graph.num_nodes))
                for r in l:
                    deck_nodes.remove(r)
                edge_index, edge_attr = subgraph( subset=deck_nodes, edge_index=graph.edge_index, edge_attr=graph.edge_attr, relabel_nodes=True, num_nodes=graph.num_nodes )
                graph_subgraphs.append( Data(x=graph.x[deck_nodes], edge_index=edge_index, edge_attr=edge_attr) )
        else:
            graph_subgraphs = [Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr)]
        subgraphs.append(graph_subgraphs)
    return subgraphs


for _ in range(5):

    lr = 0.001
    model = NetSubgraphGCN( node_size, edge_size, hidden_size, 2, k )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    def train():
        model.train()
        loss_all = 0
        for i in range(0, len(train_dataset), batch_size):
            optimizer.zero_grad()
            data = train_dataset[i:i+min([batch_size,len(train_dataset)-i])]
            subgraphs = get_subgraphs(data, train_sample_size)
            subgraph_batch = [ torch.repeat_interleave(torch.tensor([u]),len(v)).unsqueeze(-1) for u,v in enumerate(subgraphs) ]
            subgraphs = list(itertools.chain.from_iterable(subgraphs))
            weights, subgraph_batch = torch.ones((len(subgraphs),1)), torch.cat(subgraph_batch, dim=0)[:,0]
            subgraph_data = Batch().from_data_list(subgraphs)
            data = Batch().from_data_list(train_dataset[i:i+min([batch_size,len(train_dataset)-i])])
            y, subgraph_data, weights, subgraph_batch = data.y.to(device), subgraph_data.to(device), weights.to(device), subgraph_batch.to(device)
            output = F.log_softmax(model(torch.ones((subgraph_data.x.size(0),1)).to(device), subgraph_data.edge_index, torch.ones((subgraph_data.edge_index.size(1),1)).to(device), subgraph_data.batch, weights, subgraph_batch), dim=-1)
            loss = F.nll_loss(output, y)
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            #print(i/len(train_dataset))
        return loss_all / len(train_dataset)

    def test(dataset, model):
        model.eval()
        correct = 0
        for i in range(0, len(dataset), 256):
            with torch.no_grad():
                data = dataset[i:i+min([256,len(dataset)-i])]
                subgraphs = get_subgraphs(data, test_sample_size)
                subgraph_batch = [ torch.repeat_interleave(torch.tensor([u]),len(v)).unsqueeze(-1) for u,v in enumerate(subgraphs) ]
                subgraphs = list(itertools.chain.from_iterable(subgraphs))
                weights, subgraph_batch = torch.ones((len(subgraphs),1)), torch.cat(subgraph_batch, dim=0)[:,0]
                subgraph_data = Batch().from_data_list(subgraphs)
                data = Batch().from_data_list(data).to(device)
                y, subgraph_data, weights, subgraph_batch = data.y.to(device), subgraph_data.to(device), weights.to(device), subgraph_batch.to(device)
                output = F.log_softmax(model(torch.ones((subgraph_data.x.size(0),1)).to(device), subgraph_data.edge_index, torch.ones((subgraph_data.edge_index.size(1),1)).to(device), subgraph_data.batch, weights, subgraph_batch), dim=-1)
                pred = output.max(dim=1)[1]
                correct += pred.eq(y).sum().item()
        return correct/len(dataset)

    def lr_scheduler(lr, epoch, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * (0.995 ** (epoch / 5))

    stime = time.time()
    for epoch in range(1, 300):
        random.shuffle(train_dataset)
        for param_group in optimizer.param_groups:
            epoch_lr = param_group['lr']
        lr_scheduler(lr, epoch, optimizer)
        loss = train()
        test_acc = test(test_dataset, model)
        train_acc = test(train_dataset, model)
        print('Epoch: {:03d}, Train Loss: {:.7f}, Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, loss, train_acc, test_acc), "LR:\t", epoch_lr, time.time()-stime)

    print("Final Test Acc:\t", test_acc)
    results.append(test_acc)

print("Result for k =", k, torch.tensor(results).mean().item())
print("Result for k =", k,torch.tensor(results).std().item())
