__author__ = 'Silvia'

import metalearning.metafeatures_ssi
import numpy as np
import pandas as pd
import metalearning.utils
from scipy.stats import rankdata

def weighted_ranking_measure(r_real, r_pred):
    n = r_real.size
    d = np.sum([((i - j)**2)*((n-i+1)+(n-j+1)) for i, j in zip(r_real, r_pred)])
    print d
    return d

def weighted_ranking_measure_normalized(r_real, r_pred):
    n = r_real.size
    d = np.sum([((i - j)**2)*((n-i+1)+(n-j+1)) for i, j in zip(r_real, r_pred)])
    return 1-((6*d)/((n**4)+(n**3)-(n**2)-n))


def spearman_correlation_coefficient(r_real, r_pred):
    m = r_real.size
    d = sum([(i - j)**2 for i, j in zip(r_real, r_pred)])
    return 1-((6*d)/((m**3)-m))


def creating_ranking_bases(data_name, file_name):
    print("Creating Ranking bases...", data_name)

    with metalearning.utils.stopwatch() as sw:

        data = pd.read_csv(file_name)

        data = data[data.columns[1:]]

        print 'data'
        print data

        df = data

        for i in range(data.shape[0]):
            rank = rankdata(np.asarray(data.iloc[i, :]), 'average')
            df.iloc[i, :] = rank
            # print rank

        df.to_csv(data_name + '.csv', sep=',')


    time = sw.duration
    print time
    return 0


def creating_bases(data_name, row, file, pred_directory):
    print("Creating AUC bases...", data_name)

    with metalearning.utils.stopwatch() as sw:

        pred_file = pred_directory+data_name+'.csv'

        print "pred_file", pred_file

        predicted = pd.read_csv(pred_file, index_col=0)

        predicted = predicted[predicted.columns]

        print 'predicted data'
        print predicted


        for i in range(predicted.shape[0]):
            for j in range(predicted.shape[1]):
                col_name = predicted.index[i] + '_' + predicted.columns[j]
                file.loc[row, col_name] = predicted.iloc[i,j]
    time = sw.duration
    print time
    return 0


def calcule_spearman(actual, predicted, file, name):
    print("Calculating Spearman ...")

    with metalearning.utils.stopwatch() as sw:
        spearman = []

        for i in range(actual.shape[0]):
            r_actual = np.asarray(actual.iloc[i, :])
            r_pred = np.asarray(predicted.iloc[i, :])
            s = spearman_correlation_coefficient(r_actual, r_pred)
            spearman.append(s)

        file[name] = pd.Series(spearman)
        print 'm_spearman', np.mean(spearman)
    time = sw.duration
    print time
    return 0


def calcule_weighted(actual, predicted, file, name):
    print("Calculating weighted measure...")

    with metalearning.utils.stopwatch() as sw:
        w = []

        for i in range(actual.shape[0]):
            r_actual = np.asarray(actual.iloc[i, :])
            r_pred = np.asarray(predicted.iloc[i, :])
            s = weighted_ranking_measure_normalized(r_actual, r_pred)
            w.append(s)

        file[name] = pd.Series(w)
        print 'm_weighted', np.mean(w)
    time = sw.duration
    print time
    return 0