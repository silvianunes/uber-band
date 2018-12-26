__author__ = 'Silvia'

import metalearning.metafeatures_ssi
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os
import metalearning.utils
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def root_mean_squared_error(pred, target):
	assert pred.shape == target.shape, "prediction and target should be of the same shape."

	return np.sqrt(np.mean([(x[0] - x[1])**2 for x in zip(pred, target)]))


def relative_squared_error(pred, target):
    assert pred.shape == target.shape, "prediction and target should be of the same shape."
    pred_mean = np.mean(target)
    return np.sum([(x[0] - x[1]) ** 2 for x in zip(pred, target)])/np.sum([(pred_mean - x) ** 2 for x in target])


def mean_absolute_deviation(pred, target):
    print pred.shape, target.shape
    assert pred.shape == target.shape, "prediction and target should be of the same shape."
    return np.mean([abs(x[0] - x[1]) for x in zip(pred, target)])


def error_functions(target_test, pred):
    y_test = target_test

    pred = pred.reshape((-1,1))
    # print pred_method.shape

    mse = mean_squared_error(pred, y_test.T)
    print 'mse', mse
    mae = mean_absolute_error(pred, y_test.T)
    print 'mae', mae
    rmse = root_mean_squared_error(pred, y_test.T)
    print 'rmse', rmse
    rse = relative_squared_error(pred, y_test.T)
    print 'rse', rse

    return mse, mae, rmse, rse


def main_default(data_name, row, error_file):
    print("Starting metalearning...", data_name)

    with metalearning.utils.stopwatch() as sw:

        dir = 'metabase/base/'
        m_features = 'metabase/metafeatures_red.csv'

        dir_list = os.listdir(dir)

        data = pd.read_csv(m_features)
        data = data.fillna(-1)

        id = 0

        for i in range(data.shape[0]):
            if data_name == data.loc[i, "dataset"]:
                id = i
                break

        df = pd.DataFrame()

        for e in dir_list:
            if e.endswith('.csv'):
                file = dir + os.path.basename(e)

                target = np.genfromtxt(file, delimiter=',')
                target = np.asmatrix(target[1:, 1:])
                target_test = target[id, :]
                target = np.delete(target, id, 0)

                pred_default = []

                for i in range(target.shape[1]):
                    pred_default.append(np.mean(target[:, i]))

                pred_default = np.asarray(pred_default)

                print 'pred_default', os.path.basename(e)
                print pred_default

                mse, mae, rmse, rse = error_functions(target_test, pred_default)

                error_file.loc[row, "dataset"] = data_name
                error_file.loc[row, os.path.basename(e) + "_mse"] = mse
                error_file.loc[row, os.path.basename(e) + "_mae"] = mae
                error_file.loc[row, os.path.basename(e) + "_rmse"] = rmse
                error_file.loc[row, os.path.basename(e) + "_rse"] = rse

                df[os.path.basename(e)] = pd.Series(list(pred_default))

    df.to_csv(data_name + '.csv', sep=',')

    time = sw.duration
    print time
    return error_file


def main_random(data_name, row, error_file):
    print("Starting metalearning...", data_name)

    with metalearning.utils.stopwatch() as sw:

        dir = 'metabase/base/'
        m_features = 'metabase/metafeatures_red.csv'

        dir_list = os.listdir(dir)

        data = pd.read_csv(m_features)
        data = data.fillna(-1)

        id = 0

        for i in range(data.shape[0]):
            if data_name == data.loc[i, "dataset"]:
                id = i
                break

        df = pd.DataFrame()

        for e in dir_list:
            if e.endswith('.csv'):
                file = dir + os.path.basename(e)

                target = np.genfromtxt(file, delimiter=',')
                target = np.asmatrix(target[1:, 1:])
                target_test = target[id, :]
                target = np.delete(target, id, 0)

                pred_random = np.random.rand(27, 1)

                print 'pred_random', os.path.basename(e)
                print pred_random

                mse, mae, rmse, rse = error_functions(target_test, pred_random)

                error_file.loc[row, "dataset"] = data_name
                error_file.loc[row, os.path.basename(e) + "_mse"] = mse
                error_file.loc[row, os.path.basename(e) + "_mae"] = mae
                error_file.loc[row, os.path.basename(e) + "_rmse"] = rmse
                error_file.loc[row, os.path.basename(e) + "_rse"] = rse

                df[os.path.basename(e)] = pd.Series(list(pred_random))

    df.to_csv(data_name + '.csv', sep=',')

    time = sw.duration
    print time
    return error_file

def main_metalearning(data_name, row, error_file):
    print("Starting metalearning...", data_name)

    with metalearning.utils.stopwatch() as sw:

        dir = 'metabase/base/'
        m_features = 'metabase/metafeatures_red.csv'

        dir_list = os.listdir(dir)

        data = pd.read_csv(m_features)
        data = data.fillna(-1)

        id = 0

        for i in range(data.shape[0]):
            if data_name == data.loc[i, "dataset"]:
                id = i
                break

        # load predictions

        pred_file = 'predictions/meta/'+data_name+'.csv'

        print "pred_file", pred_file

        predicted = pd.read_csv(pred_file)

        predicted = predicted[predicted.columns[1:]]

        print 'predicted data'
        print predicted

        for e in dir_list:
            if e.endswith('.csv'):
                file = dir + os.path.basename(e)

                target = np.genfromtxt(file, delimiter=',')
                target = np.asmatrix(target[1:, 1:])
                target_test = target[id, :]

                pred_method = np.asarray(predicted[os.path.basename(e)])

                mse, mae, rmse, rse = error_functions(target_test, pred_method)

                print 'pred_recovery', os.path.basename(e)
                print pred_method

                error_file.loc[row, "dataset"] = data_name
                error_file.loc[row, os.path.basename(e) + "_mse"] = mse
                error_file.loc[row, os.path.basename(e) + "_mae"] = mae
                error_file.loc[row, os.path.basename(e) + "_rmse"] = rmse
                error_file.loc[row, os.path.basename(e) + "_rse"] = rse

    time = sw.duration
    print time
    return error_file

