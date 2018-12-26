import os

import pandas as pd

import numpy as np

from metalearning.validation import error_measures

import metalearner

from weights_generator import weights_generator

from metalearning.validation import ranking



def weighted():

    pred_directory = 'ranking/rank/inverse/'

    dir_list = os.listdir(pred_directory)

    actual = pd.read_csv('ranking/rank/inverse/real_rank.csv')

    file = pd.DataFrame()

    for e in dir_list:
        file_name = pred_directory + os.path.basename(e)

        predicted = pd.read_csv(file_name)

        s = ranking.calcule_weighted(actual, predicted, file, os.path.basename(e))

    file.to_csv('weighted_inverse_norm.csv', sep=',')


def spearman():

    pred_directory = 'ranking/rank/no_inverse/'

    dir_list = os.listdir(pred_directory)

    actual = pd.read_csv('ranking/rank/no_inverse/real_rank.csv')

    file = pd.DataFrame()

    for e in dir_list:
        file_name = pred_directory + os.path.basename(e)

        predicted = pd.read_csv(file_name)

        s = ranking.calcule_spearman(actual, predicted, file, os.path.basename(e))

    file.to_csv('spearman_no_inverse.csv', sep=',')


def create_auc_bases():

    directory = 'C:\Users\silvia\Documents\silvia-data\\thesis_final\hyperband-initial\\'

    pred_dir = 'predictions/meta/'

    dir_list = os.listdir(directory)

    file = pd.DataFrame()

    row = 0

    for e in dir_list:
        file_name = directory + os.path.basename(e)

        meta = ranking.creating_bases(data_name=os.path.basename(e), row=row, file=file, pred_directory=pred_dir)

        row += 1

        file.to_csv('teste_pred.csv', sep=',')

        print meta


def create_ranking_bases():

    directory = 'ranking/auc/'

    dir_list = os.listdir(directory)


    for e in dir_list:
        file_name = directory + os.path.basename(e)

        meta = ranking.creating_ranking_bases(os.path.basename(e), file_name)


def calculate_error_measures():

    directory = 'C:\Users\silvia\Documents\Silvia-data\\teste_mf\\'

    dir_list = os.listdir(directory)

    error_file = pd.DataFrame()

    row = 0

    for e in dir_list:
        file_name = directory + os.path.basename(e)

        meta = error_measures.main_default(data_name=os.path.basename(e), row=row, error_file=error_file)

        row += 1

        error_file.to_csv('default_errors.csv', sep=',')

        print meta

from workflow.utils import get_space

def weights():

    directory = 'C:\Users\silvia\Documents\Silvia-data\\teste_mf\\'

    dir_list = os.listdir(directory)

    for e in dir_list:
        file_name = directory + os.path.basename(e)

        # meta = metalearner.main(os.path.basename(e), file_name)

        # pred_file = 'predictions/meta/'+os.path.basename(e)+".csv"
        #
        # predictions = pd.read_csv(pred_file)

        weights = weights_generator(os.path.basename(e))

        space = get_space(weights)

        # T = get_params(space)


if __name__ == "__main__":
    create_auc_bases()