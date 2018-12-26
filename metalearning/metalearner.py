__author__ = 'Silvia'

import utils
import metafeatures_ssi
import metafeatures_landmarking
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os
import utils
from sklearn.model_selection import StratifiedShuffleSplit



def predict(x_train, x_test, y_train, y_test, i):
    # print 'predicting...', i, x_train.shape
    y_train.ravel()
    reg = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
    model = reg.fit(x_train, y_train)
    pred = model.predict(x_test)
    return pred[0]


def learner(target, meta_features, meta_features_test, target_test):
    predictions = []
    x_train = meta_features
    x_test = meta_features_test
    y_test = target_test

    for i in range(target.shape[1]):
        # print 'building test set...', i
        y_train = target[:, i]

        pr = predict(x_train, x_test, y_train, y_test, i)
        predictions.append(pr)

    return predictions


def get_metafeatures(file_name):
    X, y, cat = utils.load_file(file_name)

    simple = metafeatures_ssi.simple_metafeatures(X, y, cat)
    stats = metafeatures_ssi.statistical_metafeatures(X, y, cat)
    info = metafeatures_ssi.information_theoretic_metafeatures(X, y, cat)
    time = metafeatures_ssi.time_metafeatures(simple["SimpleFeatureTime"], stats["StatisticalFeatureTime"],
                                                      info["ITFeatureTime"])

    X, y, cat = utils.load_file_landmarking(file_name)

    X = np.array(X)
    y = np.array(y)

    folds_split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    folds = list(folds_split.split(X, y))

    land = metafeatures_landmarking.landmarker_metafeatures(X, y, folds)

    features = [simple.values(), stats.values(), info.values(), time.values(), land.values()]
    features = [item for sublist in features for item in sublist]

    features = pd.DataFrame(features)
    features = features.replace(-np.inf, np.nan)
    features = features.fillna(-1)
    features = features[0].tolist()

    return features


def main(file_name, data_name):
    print("Starting metalearning...", data_name)

    with utils.stopwatch() as sw:

        dir = 'C:\Users\silvia\PycharmProjects\\auto-band1.4_reg\metalearning\metabase\\performance\\'
        m_features = 'C:\Users\silvia\PycharmProjects\\auto-band1.4_reg\metalearning\metabase\\metafeatures_complete.csv'
        dir_list = os.listdir(dir)

        data = pd.read_csv(m_features)
        data = data.fillna(-1)

        id = 0

        for i in range(data.shape[0]):
            if data_name == data.loc[i, "dataset"]:
                id = i
                break

        meta = get_metafeatures(file_name)

        meta_features = np.array(data)
        meta_features = np.asmatrix(meta_features[:, 1:])
        meta_features = np.delete(meta_features, id, 0)

        # names = pd.read_csv('C:\Users\silvia\PycharmProjects\\auto-band1.4_reg\metalearning\metabase\\names.csv')

        df = pd.DataFrame()

        # print df

        for e in dir_list:
            if e.endswith('.csv'):
                try:
                    file = dir + os.path.basename(e)

                    target = np.genfromtxt(file, delimiter=',')
                    target = np.asmatrix(target[1:, 1:])
                    target_test = target[id, :]
                    target = np.delete(target, id, 0)

                    r = learner(target, meta_features, meta, target_test)

                    # print r

                    df[os.path.basename(e)] = pd.Series(r)

                    # print df
                except AttributeError:
                   pass

        # df = df.set_index(names.T.iloc[:, 0])

        # df['names'] = names.T.iloc[:, 0]

        # df = df.reindex(columns=['names'] + list(df.columns[:-1]))
        # print df
        # df.to_csv(data_name+'.csv', sep=',')

    time = sw.duration
    # print time

    return df, time

