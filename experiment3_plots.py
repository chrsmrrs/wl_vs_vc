import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_list = ['ENZYMES','PROTEINS', 'NCI1', 'PTC_MR']
dataset_name = dataset_list[0] #select dataset
wl_colors = np.loadtxt("colors/wl_"+dataset_name+"_hd32.txt")
data_all = pd.read_csv('datasets/all_data_'+dataset_name)
n_layers=data_all['layer'].max()
data_stats = pd.DataFrame(columns=['max_diff','min_diff','avg_diff','layer'])
for l in range(1,n_layers+1):
    temp = data_all.loc[data_all['layer']==l]
    data_stats = data_stats.append({'max_diff':temp['diff'].max(), 'min_diff':temp['diff'].min(), 'avg_diff':temp['diff'].mean(), 'layer':l}, ignore_index=True)


avg_diff = data_stats['avg_diff'].to_numpy()

yerr = np.stack([np.array(data_stats['avg_diff']-data_stats['min_diff']), np.array(data_stats['max_diff']-data_stats['avg_diff'])],axis=0)


double_y_axis = True #True:y-axis plot; False: x-axis plot (for colors)

if double_y_axis:
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.75)
    #ax.title(dataset_name)

    twin1 = ax.twinx()


    # Offset the right spine of twin2.  The ticks and label have already been
    # placed on the right by twinx above.
    twin1.spines.right.set_position(("axes", 1.2))

    p1 = ax.errorbar(np.array([i for i in range(1,n_layers+1)]),avg_diff, yerr=yerr, fmt='.k')
    p2, = twin1.plot(np.array([i for i in range(1,n_layers+1)]), wl_colors, "r*-", label="Colors")
    ax.set_xlim(0, data_stats['layer'].max()+1)
    ax.set_ylim(0, data_stats['max_diff'].max()+2)
    twin1.set_ylim(np.min(wl_colors)-10, np.max(wl_colors)+1000)


    ax.set_xlabel("Layers")
    ax.set_ylabel('train_error - test_error')
    ax.set_title(dataset_name)
    twin1.set_ylabel("Colors")
    twin1.yaxis.label.set_color(p2.get_color())

    tkw = dict(size=4, width=1.5)

    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)

    #ax.legend(handles=[p1, p2])

    plt.show()

    plt.savefig('relu_'+dataset_name+'_errorbars_colors_yaxis')
else:
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.75)
    #ax.title(dataset_name)

    twin1 = ax.twiny()

    p1 = ax.errorbar(np.array([i for i in range(1,11)]),avg_diff, yerr=yerr, fmt='.k')

    ax.set_xlabel("Layers")
    twin1.set_xlabel("Colors")
    ax.set_ylabel('train_error - test_error')
    ax.set_title(dataset_name)

    ax.set_xlim(0, data_stats['layer'].max()+1)
    twin1.set_xlim(np.min(wl_colors)-20,np.max(wl_colors)+100)
    ax.set_ylim(0, data_stats['max_diff'].max()+2)
    plt.savefig('relu_'+dataset_name+'_errorbars_colors_xaxis')