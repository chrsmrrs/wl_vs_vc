import os.path as osp

import numpy as np
from wl.wl import wl_simple_color_count

num_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
dataset = "PROTEINS"


num_colors = wl_simple_color_count(dataset, max(num_layers), degree=False, uniform=False)

print(num_colors)

