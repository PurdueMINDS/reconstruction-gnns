import torch, copy, random, time
from datasets_generation.build_cycles import FourCyclesDataset
from layers.gnn import *
import torch.nn.functional as F
from torch_geometric.utils import degree


rootdir = './data/datasets_kcycle_nsamples=10000/'

""" Interactive Shell Command: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=urca-gpu --time=12:00:00 --pty bash """

""" Run: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=ml-all-gpu --output=pna-4.txt python -u pna.py 4 36 & """
""" Run: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=ml-all-gpu --output=pna-6.txt python -u pna.py 6 56 &  """
""" Run: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=ml-all-gpu --output=pna-8.txt python -u pna.py 8 72 & """

# nohup python -u pna.py 4 36 > pna-4.txt &
# nohup python -u pna.py 6 56 > pna-6.txt &
# nohup python -u pna.py 8 72 > pna-8.txt &

k = int(sys.argv[1])
graph_size = int(sys.argv[2])

#k_lst=[4, 6, 8]
#graph_size_lst=[36, 56, 72]

train_dataset = list(FourCyclesDataset(k, graph_size, rootdir, proportion=1.0, train=True, transform=None))
test_dataset = list(FourCyclesDataset(k, graph_size, rootdir, proportion=1.0, train=False, transform=None))


device = torch.device("cuda:0")

node_size = 1
edge_size = 1
batch_size = 16

results = []

for _ in range(5):

    lr = 0.001
    deg = torch.zeros(10, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    model = NetPNA( node_size, edge_size, 2, deg, k )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    def train():
        model.train()
        loss_all = 0
        lf = torch.nn.NLLLoss()
        for i in range(0, len(train_dataset), batch_size):
            optimizer.zero_grad()
            data = train_dataset[i:i+min([batch_size,len(train_dataset)-i])]
            data = Batch().from_data_list(data).to(device)
            output = F.log_softmax(model(torch.zeros(data.x.size(0)).to(device).long(), data.edge_index, torch.zeros((data.edge_index.size(1),1)).to(device).long(), data.batch), dim=-1)
            loss = lf(output, data.y)
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
        return loss_all / len(train_dataset)

    def test(dataset, model):
        model.eval()
        correct = 0
        for i in range(0, len(dataset), batch_size):
            with torch.no_grad():
                data = dataset[i:i+min([batch_size,len(dataset)-i])]
                data = Batch().from_data_list(data).to(device)
                output = F.log_softmax(model(torch.zeros(data.x.size(0)).to(device).long(), data.edge_index, torch.zeros((data.edge_index.size(1),1)).to(device).long(), data.batch), dim=-1)
                pred = output.max(dim=1)[1]
                correct += pred.eq(data.y).sum().item()
        return correct/len(dataset)

    def lr_scheduler(lr, epoch, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * (0.9 ** (epoch / 5))

    stime = time.time()
    for epoch in range(1, 100):
        random.shuffle(train_dataset)
        for param_group in optimizer.param_groups:
            epoch_lr = param_group['lr']
        lr_scheduler(lr, epoch, optimizer)
        loss = train()
        test_acc = test(test_dataset, model)
        train_acc = test(train_dataset, model)
        print('Epoch: {:03d}, Train Loss: {:.7f}, Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, loss, train_acc, test_acc), "LR:\t", epoch_lr, "Time:\t", time.time()-stime)

    print("Final Test Acc:\t", test_acc)
    results.append(test_acc)

print("Result for k =", k, torch.tensor(results).mean().item())
print("Result for k =", k,torch.tensor(results).std().item())
