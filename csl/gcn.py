import torch, copy
from torch_geometric.datasets import GNNBenchmarkDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from layers.gnn import *
from data_utils import *
from idxs import *

""" Interactive Shell Command: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=copa-gpu --time=12:00:00 --pty bash """

""" Run: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=copa-gpu --output=gcn.txt python -u gcn.py & """

path = "."
node_size = 1
edge_size = 1
hidden_size = 110
batch_size = 5
device = torch.device("cuda:0")

dataset = GNNBenchmarkDataset(path, name="CSL", pre_transform=CSLPreTransform()).shuffle()

result = []

for train, val, test in zip(train_idx, val_idx, test_idx):

    train_dataset = dataset[train]
    val_dataset = dataset[val]
    test_dataset = dataset[test]

    model = NetGCN( node_size, edge_size, hidden_size, dataset.num_classes )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2*5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    def train():
        model.train()
        loss_all = 0
        lf = torch.nn.CrossEntropyLoss()
        for i in range(0, len(train_dataset), batch_size):
            optimizer.zero_grad()
            data = train_dataset[i:i+min([batch_size,len(train_dataset)-i])]
            data = Batch().from_data_list(data).to(device)
            loss = lf(model(data.x_, data.edge_index, data.edge_attr_, data.batch), data.y)
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
        return loss_all / len(train_dataset)

    def test(dataset, model):
        model.eval()
        correct = 0
        for i in range(0, len(dataset), 256):
            data = dataset[i:i+min([256,len(dataset)-i])]
            data = Batch().from_data_list(data).to(device)
            output = model(data.x_, data.edge_index, data.edge_attr_, data.batch)
            pred = output.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
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
    print("Test Acc:\t", tests[best_val_epoch])
    result.append(tests[best_val_epoch])

print( "Result:\t", np.array(result).mean(), np.array(result).std() )
