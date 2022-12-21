import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx


def random_tree(order):
    # build edges from depth 1 to root
    num_depth_1 = int(np.sqrt(order))
    src_depth_1 = torch.arange(1, num_depth_1 + 1)
    tgt_depth_1 = torch.zeros((num_depth_1,), dtype=torch.long)
    edges_depth_1 = torch.stack([src_depth_1, tgt_depth_1], dim=0)

    # sample edges from depth 2 to depth 1
    num_depth_2 = order - num_depth_1 - 1
    src_depth_2 = torch.arange(num_depth_1 + 1, order)
    tgt_depth_2 = torch.randint(1, num_depth_1 + 1, (num_depth_2,))
    edges_depth_2 = torch.stack([src_depth_2, tgt_depth_2], dim=0)

    edge_index = torch.cat([edges_depth_1, edges_depth_2], dim=1)

    x = torch.ones((order,), dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index)
    return data


def get_wl_hash(data):
    G = to_networkx(data)
    wl_hash = nx.algorithms.weisfeiler_lehman_graph_hash(G)
    return wl_hash


def get_unique_trees(order, num_trees):
    data_list = []
    wl_hashes = set()

    tries = 0
    while len(data_list) < num_trees:
        data = random_tree(order)
        wl_hash = get_wl_hash(data)
        if wl_hash not in wl_hashes:
            data_list.append(data)
            wl_hashes |= {wl_hash}
        tries += 1
        if tries > 10 * num_trees:
            raise RuntimeError(f'Could not find {num_trees} trees of order {order}')
    return data_list
