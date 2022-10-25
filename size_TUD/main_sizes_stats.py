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

dataset_name_list = ["Mutagenicity", "NCI1", "NCI109",  "MCF-7", "MCF-7H"]

num_reps = 5
ratios = [(0.6, 0.7), (.5,.7), (.5,.8), (0.4, 0.9)]



def split_dataset(dataset, low, up):
    sizes = []
    for data in dataset:
        sizes.append(data.num_nodes)

    l = np.quantile(sizes, low)
    u = np.quantile(sizes, up)

    lower_indices = []
    upper_indices = []
    for i, data in enumerate(dataset):
        if data.num_nodes <= l:
            lower_indices.append(i)
        if data.num_nodes >= u:
            upper_indices.append(i)

    return (lower_indices, upper_indices)

for dataset_name in dataset_name_list:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
    dataset = TUDataset(path, name=dataset_name).shuffle()

    print(dataset_name)

    for i, (low, up) in enumerate(ratios):

        print(low, up)

        dataset.shuffle()
        lower_indices, upper_indices = split_dataset(dataset, low, up)

        train_dataset = dataset[lower_indices]
        test_dataset = dataset[upper_indices]

        print(dataset.data.y[lower_indices].sum() / dataset.data.y[lower_indices].size(-1))
        print(dataset.data.y[upper_indices].sum() / dataset.data.y[upper_indices].size(-1))

        print(len(train_dataset), len(test_dataset))


