import torch
from torch.nn import Module, Linear, Sequential, BCEWithLogitsLoss, ModuleList, LeakyReLU, ReLU, GELU, Dropout
from torch_geometric.nn import GraphConv
from torch_geometric.nn.pool import global_add_pool, global_mean_pool
from tqdm import trange


class GNNLayer(Module):

    def __init__(self, in_dim, out_dim, dropout_p=0.0):
        super(GNNLayer, self).__init__()
        self.conv = GraphConv(in_channels=in_dim, out_channels=out_dim)
        self.act = ReLU(inplace=True)
        self.dropout = Dropout(dropout_p)

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)
        out = self.act(out)
        out = self.dropout(out)
        return out


class GNN(Module):

    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, pool_fn='add', **kwargs):
        super(GNN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        in_dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        out_dims = [hidden_dim] * num_layers

        layers = []
        for in_dim, out_dim in zip(in_dims, out_dims):
            layers.append(GNNLayer(in_dim=in_dim, out_dim=out_dim))
        self.layers = ModuleList(layers)

        self.pool_fn = global_mean_pool if pool_fn == 'mean' else global_add_pool

        self.graph_cls = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(inplace=True),
            Linear(hidden_dim, self.out_dim),
        )

        self.criterion = BCEWithLogitsLoss()

    def forward(self, data):
        x = data.x.view(-1, self.in_dim)
        for layer in self.layers:
            x = layer(x, data.edge_index)

        x = self.pool_fn(x, data.batch)
        y = self.graph_cls(x)
        return y

    def epoch(self, data, opt, steps, tqdm_prefix='', **kwargs):
        self.train()

        best_acc = 0.0
        with trange(steps) as t:
            t.set_description(tqdm_prefix)
            for s in t:
                opt.zero_grad()

                y_pred = self(data)
                y_true = data.y.view(-1, self.out_dim)
                loss = self.criterion(y_pred, y_true)

                loss.backward()
                opt.step()

                rounded = (y_pred > 0.0).float()

                acc = (rounded == y_true).float().mean().cpu().item()
                loss = loss.cpu().item()
                t.set_postfix(loss=loss, acc=acc)
                if acc > best_acc:
                    best_acc = acc
                if acc == 1.0:
                    break
        return acc

    def evaluate(self, data):
        self.eval()
        with torch.no_grad():
            y_pred = self(data)
            y_true = data.y.view(-1, self.out_dim)

            rounded = (y_pred > 0.0).float()
            acc = (rounded == y_true).float().mean().cpu().item()
        return acc

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return opt

    def fit(self, data, train_steps, lr=0.001, weight_decay=1e-5, device='cuda:0', **kwargs):
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(device)
        data.to(device)
        acc = self.epoch(data, opt, train_steps, **kwargs)
        return acc
