from argparse import ArgumentParser
import numpy as np
import pandas as pd
import os

import torch
from torch_geometric.data import Batch

from model import GNN
from data import get_tnm_trees


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--num_layers", type=int, default=[3], nargs='+', help="Number of layers")
    parser.add_argument("--hidden_dim", type=int, default=[4, 16, 64, 256], nargs='+', help="Hidden Dimension")
    parser.add_argument("--lr", type=float, default=1.e-4, help="Learning Rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight Decay")
    parser.add_argument("--pool_fn", type=str, default='sum', help="Pool")
    parser.add_argument("--order", type=int, default=[10, 20, 30, 40, 50, 60, 70, 80, 90], nargs='+', help="Order of trees")
    parser.add_argument("--num_assign", type=int, default=50, help="Number of assignments to try per set of trees")
    parser.add_argument("--train_steps", type=int, default=100000, help="Number of steps per epoch")
    parser.add_argument("--file_name", type=str, default='simple_tree_results.csv', help="CSV file to store results")
    parser.add_argument("--seed", type=int, default=6, help="the random seed for torch and numpy")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    kwargs = vars(args)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    csv_path = os.path.join('./results', f'{args.file_name}')
    os.makedirs('./results', exist_ok=True)
    results = []

    for o in args.order:
        data_list = get_tnm_trees(o - 3)
        data = Batch.from_data_list(data_list)
        for a in range(args.num_assign):
            data.y = torch.randint(0, 2, (len(data_list),), device=data.x.device)

            for h in args.hidden_dim:
                for l in args.num_layers:
                    model = GNN(in_dim=1, out_dim=1, hidden_dim=h, num_layers=l, pool_fn=args.pool_fn)

                    acc = model.fit(
                        data=data,
                        device=device,
                        tqdm_prefix=f'h={h},l={l},o={o},a={a}',
                        **kwargs
                    )

                    result_record = {
                        'order': o,
                        'assignment': a,
                        'hidden_dim': h,
                        'layers': l,
                        'accuracy': acc
                    }
                    results.append(result_record)
                    df = pd.DataFrame(results)
                    df.to_csv(csv_path)
