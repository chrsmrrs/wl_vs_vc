import os.path as osp
import csv

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, global_add_pool, GraphConv
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math as m

batch_size = 128
num_layers = [0, 1, 2, 3, 4, 5, 6]
lr = 0.001
epochs = 500
#dataset_name_list = ["MCF-7", "MCF-7H"]
dataset_name_list = ["AIDS",]
num_reps = 5

# color_counts = [
#     [3, 231, 10416, 15208, 16029, 16450, 16722, 16895, 17026],
#     [14, 274, 4327, 18309, 38013, 55650, 68257, 76872, 82412],
#     [37, 292, 4058, 22948, 44508, 58948, 68632, 75754, 81263],
#     [38, 283, 4098, 23411, 45045, 59454, 69155, 76292, 81744],
#     [46, 487, 9543, 78604, 188976, 284930, 361501, 422537, 469318],

# ]



color_counts = [
    [1015, 1849, 1901, 1902, 1902, 1902, 1902, 1902, 1902],
]


hd = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            x = torch.relu(conv(x, edge_index))
        x = global_add_pool(x, batch)

        return self.mlp(x)


for d, dataset_name in enumerate(dataset_name_list):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
    dataset = TUDataset(path, name=dataset_name).shuffle()

    colors = sns.color_palette()  # ["darkorange", "royalblue", "darkorchid", "limegreen"]

    raw_data = []
    table_data = []

    diffs = []
    diffs_std = []

    for l in num_layers:
        print(l)
        table_data.append([])
        for it in range(num_reps):

            dataset.shuffle()

            train_dataset = dataset[len(dataset) // 10:]
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

            test_dataset = dataset[:len(dataset) // 10]
            test_loader = DataLoader(test_dataset, batch_size)

            model = Net(dataset.num_features, hd, dataset.num_classes, l).to(device)
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

            raw_data.append({'it': it, 'test': test_acc, 'train': train_acc, 'diff': train_acc - test_acc, 'layer': l, 'Color classes': color_counts[d][l]})


            table_data[-1].append([train_acc, test_acc, train_acc - test_acc, color_counts[d][l]])

    data = pd.DataFrame.from_records(raw_data)

    # ax = sns.pointplot(x='layer',
    #                    y='diff', linestyles='',
    #                   data=data, color=colors[0], )
    #
    # sns.lineplot(x='layer', y='Color classes', data=data, color=colors[1], ax=ax.axes.twinx())
    #
    # ax.set(title=dataset_name, xlabel='Layer', ylabel='Train - test accuracy [%]')
    # plt.legend(loc='lower right', labels=['Color classes'])
    #
    # #ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    # plt.tight_layout()
    # plt.savefig("colors_" + str(dataset_name) + ".pdf")
    # plt.show()
    #
    #
    # plt.close()



    print("#####")
    print(dataset_name)
    print("#####")

    table_data = np.array(table_data)

    with open(dataset_name + '.csv', 'w') as file:
        writer = csv.writer(file, delimiter=' ', lineterminator='\n')

        for i, h in enumerate(num_layers):
            train = table_data[i][:, 0]
            test = table_data[i][:, 1]
            diff = table_data[i][:, 2]
            color = table_data[i][:, 3]

            writer.writerow([str(h)])
            writer.writerow(["###"])
            writer.writerow([train.mean(), train.std()])
            writer.writerow([test.mean(), test.std()])
            writer.writerow([diff.mean(), diff.std()])
            #writer.writerow([color[-1]])

            print(str(h))
            print("###")
            print(train.mean(), train.std())
            print(test.mean(), test.std())
            print(diff.mean(), diff.std())
            print(color[-1])

    # data = pd.DataFrame.from_records(raw_data)
    # data.to_csv(dataset_name + '_relu')
