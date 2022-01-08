import torch, copy, random, itertools, time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from layers.gnn import *
from data_utils import *
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from random import sample
import pickle
from torch_geometric.utils import degree

import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

""" Interactive Shell Command: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=ml-all-gpu --time=12:00:00 --pty bash """

""" Run: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=ml-all-gpu --output=deck1pna.txt python -u deck-n-l-pna.py 1 & """
""" Run: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=ml01-gpu,ml02-gpu,ml03-gpu,ml04-gpu,ml05-gpu,ml06-gpu,ml07-gpu,ml08-gpu --output=deck2pna.txt python -u deck-n-l-pna.py 2 & """
""" Run: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=ml-all-gpu --output=deck3pna.txt python -u deck-n-l-pna.py 3 & """
""" Run: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=ml-all-gpu --output=deckhalfpna.txt python -u deck-n-l-pna.py half & """

# nohup python -u deck-n-l-pna.py 1 > deck1pna.txt &
# nohup python -u deck-n-l-pna.py 2 > deck2pna.txt &
# nohup python -u deck-n-l-pna.py 3 > deck3pna.txt &
# nohup python -u deck-n-l-pna.py half > deckhalfpna.txt &

import math
ell = -1
if sys.argv[1].isdigit(): ell = int(sys.argv[1])

device = torch.device("cuda:1")
with open("multitask_dataset.pkl", 'rb') as f:
    (adj, features, node_labels, graph_labels) = pickle.load(f)
dataset = to_torch_geom(adj, features, node_labels, graph_labels, device, False)

path = "."
node_size = 1
edge_size = 1
hidden_size = 300
batch_size = 16
train_sample_size = 15
test_sample_size = 10

train_dataset = [graph for size in dataset["train"] for graph in size]
val_dataset = [graph for size in dataset["val"] for graph in size]
test_dataset = [graph for size in dataset["test"] for graph in size]

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

    deg = torch.zeros(25, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    model = NetSubgraphPNA( node_size, edge_size, 3, deg)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    def train():
        model.train()
        loss_all = 0
        lf_reg = torch.nn.MSELoss()
        #lf_clf = torch.nn.CrossEntropyLoss()
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
            output = model(torch.zeros(subgraph_data.x.size(0)).to(device).long(), subgraph_data.edge_index, torch.zeros(subgraph_data.edge_index.size(1)).to(device).long(), subgraph_data.batch, weights, subgraph_batch)
            loss_reg = lf_reg(output, y)
            #loss_clf = lf_clf(output[:,:2], data.y[:,0].long())
            loss = loss_reg
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
        return loss_all / len(train_dataset)

    def test(dataset, model):
        model.eval()
        error = 0
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
                output = model(torch.zeros(subgraph_data.x.size(0)).to(device).long(), subgraph_data.edge_index, torch.zeros(subgraph_data.edge_index.size(1)).to(device).long(), subgraph_data.batch, weights, subgraph_batch)
                error += ((y - output) ** 2).sum(dim=0)
        error_log = torch.log(error/len(dataset))
        return error_log.detach().cpu()

    valids = []
    tests = []

    stime = time.time()
    for epoch in range(1, 50000000):
        random.shuffle(train_dataset)
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train()
        val_mae = test(val_dataset, model)
        test_mae = test(test_dataset, model)
        train_mae = test(train_dataset, model)
        scheduler.step(val_mae.mean().item())
        valids.append(val_mae.mean().item())
        tests.append(test_mae)
        print(epoch, lr, loss, train_mae, val_mae, test_mae, time.time()-stime)
        if lr < 1e-6:
                print("Converged.")
                break
    best_val_epoch = np.argmin(np.array(valids))
    print("Test MAE:\t", tests[best_val_epoch])
    results.append(tests[best_val_epoch].unsqueeze(0))

print(torch.cat( results, dim=0 ).mean(dim=0) )
print(torch.cat( results, dim=0 ).std(dim=0) )
