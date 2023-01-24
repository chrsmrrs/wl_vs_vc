import numpy as np
import torch
from torch_geometric.data import Data


def tnm_tree(deg_1, deg_2):
    # build edges from depth 1 to root
    src_depth_1 = torch.arange(1, 3)
    tgt_depth_1 = torch.zeros((2,), dtype=torch.long)
    edges_depth_1 = torch.stack([src_depth_1, tgt_depth_1], dim=0)

    # sample edges from depth 2 to depth 1
    num_depth_2 = deg_1 + deg_2
    order = num_depth_2 + 3
    src_depth_2 = torch.arange(3, order)
    tgt_depth_2 = torch.ones((num_depth_2,), dtype=torch.long)
    tgt_depth_2[deg_1:] = 2

    edges_depth_2 = torch.stack([src_depth_2, tgt_depth_2], dim=0)

    edge_index = torch.cat([edges_depth_1, edges_depth_2], dim=1)

    x = torch.ones((order,), dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index)
    return data


def get_tnm_trees(order):
    data_list = []
    max_idx = int(np.floor(order / 2))
    for i in range(max_idx+1):
        deg_1 = i
        deg_2 = order - i
        data = tnm_tree(deg_1, deg_2)
        data_list.append(data)
    return data_list
