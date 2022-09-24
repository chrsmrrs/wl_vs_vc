import os.path as osp

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, global_add_pool, GraphConv

batch_size = 128
num_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
lr = 0.001
epochs = 500
datasets = ["ENZYMES", "MCF-7", "MOLT", "Mutagenicity", "NCI1", "NCI109"]
num_reps = 10

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


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')

raw_data = []
table_data = []
for dataset in datasets:
    dataset_name = dataset
    dataset = TUDataset(path, name=dataset).shuffle()

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

            raw_data.append({'test': test_acc, 'train': train_acc, 'diff': train_acc - test_acc, 'it': it, 'layer': l})

    data = pd.DataFrame.from_records(raw_data)
    data.to_csv(dataset_name + '_relu')
