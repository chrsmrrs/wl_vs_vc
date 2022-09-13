import csv
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, global_add_pool
import torch_geometric.transforms as T
from conv import GraphConv
from torch_geometric.utils import degree


batch_size = 128
num_layers = 5
lr = 0.001
epochs = 500
dataset_name_list = ["IMDB-BINARY", "IMDB-BINARY", "ENZYMES", "PTC_MR", "NCI1"]
#dataset_name_list = ["ENZYMES"]
num_reps = 10
hds = [16, 64, 256, 1024]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return


# Simple GNN layer from paper.
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, nc):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(nc):
            self.convs.append(GraphConv(in_channels, hidden_channels, aggr='add', bias=True))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels])

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)

        return self.mlp(x)


for dataset_name in dataset_name_list:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
    dataset = TUDataset(path, name=dataset_name).shuffle()

    # One-hot degree if node labels are not available.
    # The following if clause is taken from  https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/datasets.py.
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)


    colors = ["darkorange", "royalblue", "darkorchid", "red"]

    raw_data = []
    table_data = []

    for i, hc in enumerate(hds):
        table_data.append([])
        for it in range(num_reps):

            dataset.shuffle()

            train_dataset = dataset[len(dataset) // 10:]
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

            test_dataset = dataset[:len(dataset) // 10]
            test_loader = DataLoader(test_dataset, batch_size)

            model = Net(dataset.num_features, hc, dataset.num_classes, num_layers).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)


            def train():
                model.train()

                total_loss = 0
                for data in train_loader:
                    data = data.to(device)
                    optimizer.zero_grad()
                    out = model(data.x, data.edge_index, data.batch)
                    loss = F.cross_entropy(out, data.y)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss) * data.num_graphs
                return total_loss / len(train_loader.dataset)


            @torch.no_grad()
            def test(loader):
                model.eval()

                total_correct = 0
                for data in loader:
                    data = data.to(device)
                    pred = model(data.x, data.edge_index, data.batch).argmax(dim=-1)
                    total_correct += int((pred == data.y).sum())
                return total_correct / len(loader.dataset)


            for epoch in range(1, epochs + 1):
                loss = train()
                train_acc = test(train_loader) * 100.0
                test_acc = test(test_loader) * 100.0

                # print(it, epoch, train_acc, test_acc, train_acc - test_acc)
                raw_data.append(
                    {'epoch': epoch, 'test': test_acc, 'train': train_acc, 'diff': train_acc - test_acc, 'it': it,
                     'hidden_channels': hc})

            table_data[-1].append([train_acc, test_acc, train_acc - test_acc])

        data = pd.DataFrame.from_records(raw_data)
        data = data.astype({'epoch': int})

        ax = sns.lineplot(x='epoch',
                          y='train',
                          data=data, alpha=1.0, color=colors[i], linestyle='--')

        ax = sns.lineplot(x='epoch',
                          y='test',
                          data=data, alpha=1.0, color=colors[i])

        # ax = sns.lineplot(x='epoch',
        #                   y='diff',
        #                   data=data, color=colors[i], linestyle='--')

        ax.set(xlabel='Epoch', ylabel='Accuracy [%]')

    table_data = np.array(table_data)

    temp = []

    print("#####")
    print(dataset_name)
    print("#####")

    with open(dataset_name + '.csv', 'w') as file:
        writer = csv.writer(file, delimiter=' ', lineterminator='\n', )

        for i, h in enumerate(hds):
            train = table_data[i][:, 0]
            test = table_data[i][:, 1]
            diff = table_data[i][:, 2]

            writer.writerow([str(h)])
            writer.writerow(["###"])
            writer.writerow([train.mean(), train.std()])
            writer.writerow([test.mean(), test.std()])
            writer.writerow([diff.mean(), diff.std()])

            print(str(h))
            print("###")
            print(train.mean(), train.std())
            print(test.mean(), test.std())
            print(diff.mean(), diff.std())

    plt.savefig("weights_" + str(dataset_name) + ".pdf")
    plt.show()
