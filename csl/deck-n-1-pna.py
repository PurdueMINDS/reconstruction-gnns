import torch, time, itertools, random, copy
from torch_geometric.datasets import GNNBenchmarkDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from layers.gnn import *
from data_utils import *
from external_libs.subgraph_utils import *
from idxs import *

""" Interactive Shell Command: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=ml-all-gpu --time=12:00:00 --pty bash """

""" Run: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=ml-all-gpu --output=deck1pna.txt python -u deck-n-1-pna.py & """

path = "."
node_size = 2
edge_size = 2
hidden_size = 110
batch_size = 5
sample_training = False
sample_size = 20
device = torch.device("cuda:0")

dataset = GNNBenchmarkDataset(path, name="CSL", pre_transform=CSLPreTransform()).shuffle()

result = []

for train, val, test in zip(train_idx, val_idx, test_idx):

    train_dataset = dataset[train]
    val_dataset = dataset[val]
    test_dataset = dataset[test]

    deg = torch.zeros(5, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    model = NetSubgraphPNA( node_size, edge_size, dataset.num_classes, deg )
    model = model.to(device)
    model = model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr=2*5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    def train():
        model.train()
        loss_all = 0
        lf = torch.nn.CrossEntropyLoss()
        j = 0
        stime = time.time()

        for i in range(0, len(train_dataset), batch_size):
            j+=1
            optimizer.zero_grad()
            data = Batch().from_data_list(train_dataset[i:i+min([batch_size,len(train_dataset)-i])])
            if sample_training:
                subgraphs = [ random.sample(graph.deck, min([sample_size, len(graph.deck) ])) for graph in train_dataset[i:i+min([batch_size,len(train_dataset)-i])] ]
            else:
                subgraphs = [ graph.deck for graph in train_dataset[i:i+min([batch_size,len(train_dataset)-i])] ]
            subgraph_batch = [ torch.repeat_interleave(torch.tensor([u]),len(v)).unsqueeze(-1) for u,v in enumerate(subgraphs) ]
            subgraphs = list(itertools.chain.from_iterable(subgraphs))
            weights, subgraph_batch = torch.ones((len(subgraphs),1)), torch.cat(subgraph_batch, dim=0)[:,0]
            subgraph_data = Batch().from_data_list(subgraphs)
            y, subgraph_data, weights, subgraph_batch = data.y.to(device), subgraph_data.to(device), weights.to(device), subgraph_batch.to(device)
            loss = lf( model(subgraph_data.x.long(), subgraph_data.edge_index, subgraph_data.edge_attr.long(), subgraph_data.batch, weights, subgraph_batch), y )
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
        return loss_all / len(train_dataset)

    def test(dataset, model):
        model.eval()
        correct = 0
        for i in range(0, len(dataset), batch_size):
            data_ = dataset[i:i+min([batch_size,len(dataset)-i])]
            data = Batch().from_data_list(data_)
            subgraphs = [ graph.deck for graph in data_ ]
            subgraph_batch = [ torch.repeat_interleave(torch.tensor([u]),len(v)).unsqueeze(-1) for u,v in enumerate(subgraphs) ]
            subgraphs = list(itertools.chain.from_iterable(subgraphs))
            weights, subgraph_batch = torch.ones((len(subgraphs),1)), torch.cat(subgraph_batch, dim=0)[:,0]
            subgraph_data = Batch().from_data_list(subgraphs)
            y, subgraph_data, weights, subgraph_batch = data.y.to(device), subgraph_data.to(device), weights.to(device), subgraph_batch.to(device)
            output = model(subgraph_data.x.long(), subgraph_data.edge_index, subgraph_data.edge_attr.long(), subgraph_data.batch, weights.double(), subgraph_batch)
            pred = output.max(dim=1)[1]
            correct += pred.eq(y).sum().item()
        return correct/len(dataset)

    valids = []
    tests = []

    for epoch in range(1, 50000000):
        train_dataset = train_dataset.shuffle()
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train()
        val_acc = test(val_dataset, model)
        test_acc = test(test_dataset, model)
        train_acc = test(train_dataset, model)
        scheduler.step(val_acc)
        valids.append(val_acc)
        tests.append(test_acc)
        print('Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Train Acc: {:.7f}, Validation Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, lr, loss, train_acc, val_acc, test_acc))
        if lr < 1e-6:
                print("Converged.")
                break
    best_val_epoch = np.argmax(np.array(valids))
    print("Final Test Acc:\t", tests[best_val_epoch])
    result.append(tests[best_val_epoch])

print( "Result:\t", np.array(result).mean(), np.array(result).std() )
