import Orange
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import matplotlib.patches as mpatches
import os

def plot_cd():

    names = ["HB", "ML_rank + HB", "Uber-Band"]
    avranks = [1.29070, 2.03488, 2.67442]

    cd = Orange.evaluation.scoring.compute_CD(avranks, 43, alpha="0.05", type="nemenyi")#tested on 12 datasets
    Orange.evaluation.scoring.graph_ranks('cd_valid.png',avranks, names, cd=cd, width=6, textspace=1.5, fontsize=7)
    plt.show()


def rank(data):
    print data

    df = data

    for i in range(data.shape[0]):
        rank = rankdata(np.asarray(data.iloc[i, :]), 'average')
        # rank = 4-rank
        df.iloc[i, :] = rank

    df.to_csv('rank_test.csv')

def plot_bar():
    data = pd.read_csv('C:\Users\silvia\Dropbox\UB_comparison_test_1\\benchmark_results\\results_pakdd\\auc-test.csv', index_col=0)

    rank(data)

    valid = pd.read_csv('rank_valid.csv')

    test = pd.read_csv('rank_test.csv')

    valid = valid.set_index('dataset')

    test = test.set_index('dataset')

    fig, axs = plt.subplots(2,1)

    valid.plot(kind='bar', ax=axs[0], fontsize=8)
    test.plot(kind='bar', ax=axs[1], fontsize=8)

    red_patch = mpatches.Patch(color='red', label='hyperband')
    blue_patch = mpatches.Patch(color='blue', label='uber-band')
    green_patch = mpatches.Patch(color='green', label='ML_R + hyperband')

    axs[0].grid(color='gray', linestyle='-', alpha=0.3)
    # axs[0].legend().set_visible(False)
    axs[0].legend(loc='upper right', fontsize=8)
    # axs[0].set_title('(a) Validation Performance', fontsize=12)
    axs[1].grid(color='gray', linestyle='-', alpha=0.3)
    axs[1].legend(loc='upper right', fontsize=8)
    # axs[1].legend().set_visible(False)
    # axs[1].set_title('(b) Test Performance', fontsize=12)

    # axs[0].plot(valid)
    # axs[0].set_xlim(0, 2)
    axs[0].set_xlabel('Datasets', fontsize=9)
    axs[0].set_ylabel('Average Rank (Validation)', fontsize=9)

    axs[1].set_xlabel('Datasets', fontsize=9)
    axs[1].set_ylabel('Average Rank (Test)', fontsize=9)
    axs[1].grid(True)

    # fig.legend((blue_patch, green_patch, red_patch), ('uber-band', 'ML_R + hyperband', 'hyperband'),
    #            loc='lower left', fontsize=7, ncol=3, mode="expand")
    # fig.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig('avg_rank_all.png')
    plt.show()


dir_data = 'C:\Users\silvia\Documents\silvia-data\\thesis_final\\benchmarkings\\'
dir_results = 'C:\Users\silvia\Dropbox\UB_comparison_test_1\\benchmark_results\\results_pakdd\meta\\'

def plot_lines_all(directory_data, directory_results):
    dir_d = os.listdir(directory_data)
    dir_r = os.listdir(directory_results)

    for e in dir_d:
        print('DATA:    ', os.path.basename(e))
        df = pd.DataFrame()

        for i in dir_r:

            if os.path.basename(e) in os.path.basename(i):

                try:
                    file = directory_results + os.path.basename(i)

                    data = pd.read_csv(file, index_col=0)

                    df['avg_'+os.path.basename(i)] = data.loc[:, 'average']

                except ValueError:
                    pass

        df.index.name = 'configuration'

        # style
        plt.style.use('seaborn-darkgrid')

        # create a color palette
        palette = plt.get_cmap('Set1')

        # multiple line plot
        num = 0

        for column in df:
            num += 100
            plt.plot(df.index, df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)

        # Add legend
        plt.legend(loc=2, ncol=2)

        # Add titles
        # plt.title("A (bad) Spaghetti plot", loc='left', fontsize=12, fontweight=0, color='orange')
        plt.xlabel("Configurations Evaluation")
        plt.ylabel("Average AUC")
        plt.savefig(os.path.basename(e)+'.png')
        plt.show()


def plot_line_grid():

    dir = 'C:\Users\silvia\Dropbox\UB_comparison_test_1\\benchmark_results\\results_pakdd\\average_auc\\to_plot\\'
    cmc = pd.read_csv(dir+'cmc.csv')
    cmc = cmc.set_index('configuration')

    cylinder_bands = pd.read_csv(dir+'cylinder-bands.csv')
    cylinder_bands = cylinder_bands.set_index('configuration')

    mozilla4 = pd.read_csv(dir + 'mozilla4.csv')
    mozilla4 = mozilla4.set_index('configuration')

    pc2 = pd.read_csv(dir + 'pc2.csv')
    pc2 = pc2.set_index('configuration')

    fig, axs = plt.subplots(2,2)

    cmc.plot(kind='line', ax=axs[0,0], fontsize=8)
    cylinder_bands.plot(kind='line', ax=axs[0,1], fontsize=8)
    pc2.plot(kind='line', ax=axs[1,0], fontsize=8)
    mozilla4.plot(kind='line', ax=axs[1,1], fontsize=8)

    red_patch = mpatches.Patch(color='red', label='ML_R + hyperband')
    blue_patch = mpatches.Patch(color='blue', label='hyperband')
    green_patch = mpatches.Patch(color='green', label='uber-band')

    axs[0,0].grid(color='gray', linestyle='-', alpha=0.3)
    axs[0,0].legend(loc='best', fontsize=8)
    axs[0,0].set_title('(a) cmc (OpenML dataset ID 23)', fontsize=9)

    axs[0,1].grid(color='gray', linestyle='-', alpha=0.3)
    axs[0, 1].legend(loc='best', fontsize=8)
    axs[0,1].set_title('(b) cylinder_bands (OpenML dataset ID 6332)', fontsize=9)

    axs[1,0].grid(color='gray', linestyle='-', alpha=0.3)
    axs[1, 0].legend(loc='best', fontsize=8)
    axs[1,0].set_title('(c) pc2 (OpenML dataset ID 1069)', fontsize=9)

    axs[1,1].grid(color='gray', linestyle='-', alpha=0.3)
    axs[1, 1].legend(loc='best', fontsize=8)
    axs[1,1].set_title('(d) mozilla4 (OpenML dataset ID 1046)', fontsize=9)

    # axs[0].plot(valid)
    # axs[0].set_xlim(0, 2)
    axs[0,0].set_xlabel('#configuration evaluation', fontsize=9)
    axs[0,0].set_ylabel('Average AUC', fontsize=9)

    axs[0,1].set_xlabel('#configuration evaluation', fontsize=9)
    axs[0,1].set_ylabel('Average AUC', fontsize=9)

    axs[1,0].set_xlabel('#configuration evaluation', fontsize=9)
    axs[1,0].set_ylabel('Average AUC', fontsize=9)

    axs[1,1].set_xlabel('#configuration evaluation', fontsize=9)
    axs[1,1].set_ylabel('Average AUC', fontsize=9)
    # axs[1].grid(True)

    # fig.legend((blue_patch, green_patch, red_patch), ('hyperband', 'uber-band', 'metalearning_rank + hyperband'),
    #            loc='lower left', fontsize=7, ncol=3, mode="expand")
    plt.savefig('per_data_.png')
    plt.show()


plot_bar()


