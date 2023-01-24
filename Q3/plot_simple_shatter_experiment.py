import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--num_layers", type=int, default=[3], nargs='+', help="Number of layers")
    parser.add_argument("--hidden_dim", type=int, default=[4, 16, 64, 256], nargs='+', help="Hidden Dimension")
    parser.add_argument("--file_names", type=str, nargs='+', default=['simple_trees_results.csv'],
                        help="CSV file to store results")
    parser.add_argument("--img_dir", type=str, default='results', help="Directory for saving plots")
    args = parser.parse_args()

    csv_paths = [os.path.join('./results', f'{n}') for n in args.file_names]
    df = pd.concat([pd.read_csv(p) for p in csv_paths])
    df['accuracy'] *= 100

    colors = sns.color_palette()
    plt.clf()

    for style_id, h in enumerate(args.hidden_dim):
        h_mask = df['hidden_dim'] == h
        for l in args.num_layers:
            l_mask = df['layers'] == l
            o_mask = df['order'] <= 90
            data = df[h_mask & l_mask & o_mask]

            ax = sns.lineplot(
                x='order',
                y='accuracy',
                data=data,
                alpha=1.0,
                color=colors[style_id],
                label=f'Dim={h}'
            )

            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.set(xlabel='|V|', ylabel='Accuracy [%]')

    plt.legend()
    plt.savefig(os.path.join(args.img_dir, f'plot_simple_trees.png'))
    plt.show()
