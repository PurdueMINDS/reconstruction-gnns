import torch, copy, random
from datasets_generation.build_cycles import FourCyclesDataset
from layers.gnn import *
import torch.nn.functional as F


rootdir = './data/datasets_kcycle_nsamples=10000/'

""" Interactive Shell Command: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=urca-gpu --time=12:00:00 --pty bash """

""" Run: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=urca-gpu --output=gine.txt python -u gine.py & """

k_lst=[4, 6, 8]
graph_size_lst=[36, 56, 72]

for k, graph_size in zip(k_lst, graph_size_lst):

    train_dataset = list(FourCyclesDataset(k, graph_size, rootdir, proportion=1.0, train=True, transform=None))
    test_dataset = list(FourCyclesDataset(k, graph_size, rootdir, proportion=1.0, train=False, transform=None))

    device = torch.device("cuda:0")

    node_size = 1
    edge_size = 1
    hidden_size = 300
    batch_size = 16

    results = []

    for _ in range(5):

        lr = 0.001
        model = NetGINE( node_size, edge_size, hidden_size, 2, k )
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        def train():
            model.train()
            loss_all = 0
            for i in range(0, len(train_dataset), batch_size):
                optimizer.zero_grad()
                data = train_dataset[i:i+min([batch_size,len(train_dataset)-i])]
                data = Batch().from_data_list(data).to(device)
                output = F.log_softmax(model(data.x, data.edge_index, torch.ones((data.edge_index.size(1),1)).to(device), data.batch), dim=-1)
                loss = F.nll_loss(output, data.y)
                loss.backward()
                loss_all += loss.item() * data.num_graphs
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
            return loss_all / len(train_dataset)

        def test(dataset, model):
            model.eval()
            correct = 0
            for i in range(0, len(dataset), 256):
                data = dataset[i:i+min([256,len(dataset)-i])]
                data = Batch().from_data_list(data).to(device)
                output = F.log_softmax(model(data.x, data.edge_index, torch.ones((data.edge_index.size(1),1)).to(device), data.batch), dim=-1)
                pred = output.max(dim=1)[1]
                correct += pred.eq(data.y).sum().item()
            return correct/len(dataset)

        def lr_scheduler(lr, epoch, optimizer):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (0.995 ** (epoch / 5))

        for epoch in range(1, 300):
            random.shuffle(train_dataset)
            for param_group in optimizer.param_groups:
                epoch_lr = param_group['lr']
            lr_scheduler(lr, epoch, optimizer)
            loss = train()
            test_acc = test(test_dataset, model)
            train_acc = test(train_dataset, model)
            print('Epoch: {:03d}, Train Loss: {:.7f}, Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, loss, train_acc, test_acc), "LR:\t", epoch_lr)

        print("Final Test Acc:\t", test_acc)
        results.append(test_acc)

    print("Result for k =", k, torch.tensor(results).mean().item())
    print("Result for k =", k,torch.tensor(results).std().item())
