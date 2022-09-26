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
from torch_geometric.utils import degree

from conv import GraphConv
from matplotlib.ticker import FormatStrFormatter


batch_size = 128
num_layers = 5
lr = 0.001
epochs = 500

dataset_name_list = ["ENZYMES", "Mutagenicity", "NCI1", "NCI109",  "MCF-7", "OVCAR-8"]
dataset_name_list = ["MCF-7H"]

num_reps = 5
hds = [4, 16, 256, 1024]


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

        if nc != 0:
            self.mlp = MLP([hidden_channels, hidden_channels, out_channels])
        else:
            self.mlp = MLP([in_channels, hidden_channels, out_channels])

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)

        return self.mlp(x)

for dataset_name in dataset_name_list:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
    dataset = TUDataset(path, name=dataset_name).shuffle()

    colors = sns.color_palette()  # ["darkorange", "royalblue", "darkorchid", "limegreen"]

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

                raw_data.append(
                    {'epoch': epoch, 'test': test_acc, 'train': train_acc, 'diff': train_acc - test_acc, 'it': it,
                     'hidden_channels': hc})

            table_data[-1].append([train_acc, test_acc, train_acc - test_acc])

        data = pd.DataFrame.from_records(raw_data)
        data = data.astype({'epoch': int})

        ax = sns.lineplot(x='epoch',
                          y='diff',
                          data=data, alpha=1.0, color=colors[i], label=str(hc))

        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

        ax.set(title = dataset_name, xlabel='Epoch', ylabel='Train - test accuracy [%]')


    table_data = np.array(table_data)

    temp = []

    print("#####")
    print(dataset_name)
    print("#####")

    with open(dataset_name + '.csv', 'w') as file:
        writer = csv.writer(file, delimiter=' ', lineterminator='\n')

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

    plt.legend(loc='lower right')
    plt.show()

    plt.close()
