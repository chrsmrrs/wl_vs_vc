import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--num_layers", type=int, default=[3], nargs='+', help="Number of layers")
    parser.add_argument("--hidden_dim", type=int, default=[64, 128, 256, 512], nargs='+', help="Hidden Dimension")
    parser.add_argument("--order", type=int, default=[20, 30, 40], nargs='+', help="Order of trees")
    parser.add_argument("--num_trees", type=int, default=[5, 10, 20, 30, 40, 50, 60, 70], nargs='+',
                        help="Number of trees")
    parser.add_argument("--file_name", type=str, default='results.csv', help="CSV file to store results")
    parser.add_argument("--img_dir", type=str, default='results', help="Directory for saving plots")
    args = parser.parse_args()

    csv_path = os.path.join('./results', f'{args.file_name}')
    df = pd.read_csv(csv_path)
    df['accuracy'] *= 100

    colors = sns.color_palette()

    for h in args.hidden_dim:
        h_mask = df['hidden_dim'] == h
        for l in args.num_layers:
            l_mask = df['layers'] == l

            plt.clf()
            for style_id, o in enumerate(args.order):
                o_mask = df['order'] == o
                data = df[h_mask & l_mask & o_mask]

                ax = sns.lineplot(
                    x='num_trees',
                    y='accuracy',
                    data=data,
                    alpha=1.0,
                    color=colors[style_id],
                    label=f'|V|={o}'
                )

                ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                ax.set(title=f'Dimension {h}, Layers {l}', xlabel='#Graphs', ylabel='Accuracy [%]')

            plt.legend()
            plt.savefig(os.path.join(args.img_dir, f'plot_{h}h_{l}l.png'))
