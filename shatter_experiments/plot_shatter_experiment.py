from argparse import ArgumentParser
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--num_layers", type=int, default=[3], nargs='+', help="Number of layers")
    parser.add_argument("--hidden_dim", type=int, default=[32, 64, 128, 256, 512], nargs='+', help="Hidden Dimension")
    parser.add_argument("--order", type=int, default=[16, 32, 64, 128], nargs='+', help="Order of trees")
    parser.add_argument("--num_trees", type=int, default=[4, 8, 16, 32, 64, 128], nargs='+', help="Number of trees")
    parser.add_argument("--file_name", type=str, default='results.csv', help="CSV file to store results")
    parser.add_argument("--img_dir", type=str, default='results', help="Directory for saving plots")
    args = parser.parse_args()

    csv_path = os.path.join('./results', f'{args.file_name}')
    df = pd.read_csv(csv_path)

    styles = ['-', '--', '-.', ':']
    colors = ['#00549F', '#000000', '#0098A1', '#612158']

    for h in args.hidden_dim:
        h_mask = df['hidden_dim'] == h
        for l in args.num_layers:
            l_mask = df['layers'] == l

            plt.clf()
            plt.axis([0, 130, 50.0, 105.0])
            plt.xlabel('#Graphs')
            plt.ylabel('Mean Accuracy (%)')
            plt.title(f'Dimension {h}, Layers {l}')

            for style_id, o in enumerate(args.order):
                o_mask = df['order'] == o
                x_data = []
                y_data = []
                for t in args.num_trees:
                    t_mask = df['num_trees'] == t
                    cur_df = df[h_mask & l_mask & o_mask & t_mask]
                    if cur_df.shape[0] > 0:
                        mean_acc = cur_df['accuracy'].mean() * 100
                        x_data.append(t)
                        y_data.append(mean_acc)
                if len(x_data) > 0:
                    plt.plot(x_data, y_data, label=f'|V|={o}', linestyle=styles[style_id], color=colors[style_id])
            plt.legend()
            plt.savefig(os.path.join(args.img_dir, f'plot_{h}h_{l}l.png'))
