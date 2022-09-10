import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
import pandas as pd
import seaborn as sns

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, global_add_pool
import matplotlib.pyplot as plt

from typing import Tuple, Union

from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size


class GraphConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels):
        super().__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_rel = Linear(in_channels[0], out_channels, bias=True)
        self.lin_root = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)
        out = self.lin_rel(out.)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_root(x_r.resize_(0))

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        return matmul(adj_t, x[0], reduce=self.aggr)


batch_size = 128
num_layers = 5
lr = 0.001
epochs = 500
dataset = "PROTEINS"
num_reps = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Simple GNN layer from paper.
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, nc):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(nc):
            self.convs.append(GraphConv(in_channels, hidden_channels))
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
# for hidden_channels in [8, 32, 128, 512]:

colors = ["red", "green", "blue"]
raw_data = []
for i, hc in enumerate([8, 64]):
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

            print(it ,epoch, train_acc, test_acc, train_acc - test_acc)
            raw_data.append({'epoch': epoch, 'test': test_acc, 'train': train_acc, 'diff': train_acc - test_acc, 'it': it, 'hidden_channels': hidden_channels})

    data = pd.DataFrame.from_records(raw_data)
    data = data.astype({'epoch': int})


    ax = sns.lineplot(x = 'epoch',
                 y = 'train',
                 data=data, alpha = 0.3, color = colors[i])

    ax =sns.lineplot(x = 'epoch',
                 y = 'test',
                 data=data, alpha = 1.0, color = colors[i])

    ax.set(xlabel='Epoch', ylabel='Accuracy [%]')

plt.savefig("proteins.pdf")

plt.show()

