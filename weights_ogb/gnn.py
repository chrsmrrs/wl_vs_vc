import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()


from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_sparse import matmul


class GraphConv(MessagePassing):
    def __init__(self, emb_dim,):
        super().__init__(aggr="add")

        self.lin_rel = Linear(emb_dim, emb_dim, bias=True)
        self.lin_root = Linear(emb_dim, emb_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    # TODO: Currently not taking edge embeddings into account.
    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x)
        out = self.lin_rel(out)

        x_r = x
        out += self.lin_root(x_r)

        return out

    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim):
        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        #self.drop_ratio = drop_ratio
        #self.JK = JK
        ### add residual connection or not
        #self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            self.convs.append(GraphConv(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### computing input node embedding

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            # if layer == self.num_layer - 1:
            #     # remove relu for the last layer
            #     h = F.dropout(h, self.drop_ratio, training=self.training)
            # else:
            #     h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            # if self.residual:
            #     h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        # if self.JK == "last":
        #     node_representation = h_list[-1]
        # elif self.JK == "sum":
        #     node_representation = 0
        #     for layer in range(self.num_layer + 1):
        #         node_representation += h_list[layer]

        node_representation = h_list[-1]

        return node_representation


class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer, emb_dim):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings

        self.gnn_node = GNN_node(num_layer, emb_dim)

        ### Pooling function to generate whole-graph embeddings
        #if self.graph_pooling == "sum":
        self.pool = global_add_pool
        # elif self.graph_pooling == "mean":
        #     self.pool = global_mean_pool
        # elif self.graph_pooling == "max":
        #     self.pool = global_max_pool
        # elif self.graph_pooling == "attention":
        #     self.pool = GlobalAttention(
        #         gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
        #                                     torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))
        # elif self.graph_pooling == "set2set":
        #     self.pool = Set2Set(emb_dim, processing_steps=2)
        # else:
        #     raise ValueError("Invalid graph pooling type.")

        # if graph_pooling == "set2set":
        #     self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_tasks)
        # else:
        self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)


if __name__ == '__main__':
    GNN(num_tasks=10)
