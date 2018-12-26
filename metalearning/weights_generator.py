__author__ = 'Silvia'

import utils
import pandas as pd
from metalearning.metalearner import main

def weights_generator(file_name, data_name):
    # print("Starting predictions generator...", data_name)

    with utils.stopwatch() as sw:


        # pred_file = 'metalearning/predictions/meta_rf/metafeatures_complete/' + data_name + ".csv"
        #
        # predictions = pd.read_csv(pred_file)

        predictions, time = main(file_name, data_name)

        # rows = predictions[predictions.columns[0]]

        names = pd.read_csv('C:\Users\silvia\PycharmProjects\\auto-band1.4_reg\metalearning\metabase\\names.csv')

        # predictions = predictions[predictions.columns[1:]]


        columns = list(predictions.head())

        for i in range(len(columns)):
            if columns[i].endswith('.csv'):
                columns[i] = columns[i][:-4]

        sum_of_auc = predictions.sum().sum()

        probabilities = pd.DataFrame()

        for i in range(predictions.shape[1]):
            p = []
            for j in range(predictions.shape[0]):
                prob_i_j = predictions.iloc[j, i]/sum_of_auc
                p.append(prob_i_j)
            probabilities[columns[i]] = pd.Series(p)

        # probabilities = probabilities.set_index(rows)

        probabilities = probabilities.set_index(names.T.iloc[:, 0])

    # time = sw.duration

    return probabilities, time

