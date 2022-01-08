import torch, copy, random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from layers.gnn import *
from data_utils import *
import pickle
from torch_geometric.utils import degree

""" Interactive Shell Command: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=ml-all-gpu --time=12:00:00 --pty bash """

""" Run: $ srun -n 1 --gres=gpu:1 --cpus-per-task=1 --nice=1000 --partition=ml-all-gpu --output=pna.txt python -u pna.py & """

# nohup python -u pna.py > pna.txt &

device = torch.device("cuda:0")
with open("multitask_dataset.pkl", 'rb') as f:
    (adj, features, node_labels, graph_labels) = pickle.load(f)
dataset = to_torch_geom(adj, features, node_labels, graph_labels, device, False)

path = "."
node_size = 1
edge_size = 1
hidden_size = 300
batch_size = 16

train_dataset = [graph for size in dataset["train"] for graph in size]
val_dataset = [graph for size in dataset["val"] for graph in size]
test_dataset = [graph for size in dataset["test"] for graph in size]

results = []

for _ in range(5):

    deg = torch.zeros(25, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    model = NetPNA( node_size, edge_size, 3, deg)

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
            data = Batch().from_data_list(data).to(device)
            output = model(torch.zeros((data.x.size(0),1)).to(device).long(), data.edge_index, torch.zeros(data.edge_index.size(1)).to(device).long(), data.batch)
            loss_reg = lf_reg(output, data.y)
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
                data = Batch().from_data_list(data).to(device)
                output = model(torch.zeros(data.x.size(0)).to(device).long(), data.edge_index, torch.zeros(data.edge_index.size(1)).to(device).long(), data.batch)
                error += ((data.y - output) ** 2).sum(dim=0)
        error_log = torch.log(error/len(dataset))
        return error_log.detach().cpu()

    valids = []
    tests = []

    #50000000
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
