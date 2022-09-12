import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, global_add_pool, GraphConv

import numpy as np

batch_size = 128
num_layers = [1,2,3,4,5,6,7,8,9,10]
lr = 0.001
epochs = 500
dataset = "ENZYMES"
num_reps = 5
hd = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Simple GNN layer from paper.
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, nc):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(nc):
            self.convs.append(GraphConv(in_channels, hidden_channels, aggr='add', bias=True))
            in_channels = hidden_channels

        # TODO: No dropout.
        self.mlp = MLP([hidden_channels, hidden_channels, out_channels])

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)

        return self.mlp(x)


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
dataset = TUDataset(path, name=dataset).shuffle()

colors = ["darkorange", "royalblue", "darkorchid"]

raw_data = []
table_data = []

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

            #print(it, epoch, train_acc, test_acc, train_acc - test_acc)
            #raw_data.append(
            #    {'epoch': epoch, 'test': test_acc, 'train': train_acc, 'diff': train_acc - test_acc, 'it': it,
            #     'hidden_channels': hd})

        print([train_acc, test_acc, train_acc - test_acc])
        table_data[-1].append([train_acc, test_acc, train_acc - test_acc])


a = np.array(table_data)
for i, _ in enumerate(num_layers):
    print(a[i][:,0].mean(), a[i][:,1].mean(), a[i][:,2].mean())


#     # data = pd.DataFrame.from_records(raw_data)
#     # data = data.astype({'epoch': int})
#     #
#     # ax = sns.lineplot(x = 'epoch',
#     #              y = 'train',
#     #              data=data, alpha = 1.0, color = colors[i], linestyle='--')
#     #
#     # ax = sns.lineplot(x = 'epoch',
#     #              y = 'test',
#     #              data=data, alpha = 1.0, color = colors[i])
#
#     # ax = sns.lineplot(x='epoch',
#     #                   y='diff',
#     #                   data=data, color=colors[i], linestyle='--')
#
#     ax.set(xlabel='Epoch', ylabel='Accuracy [%]')
#
# table_data = np.array(table_data)
#
#
# plt.savefig("weights_" + str(dataset) + ".pdf")
# plt.show()
