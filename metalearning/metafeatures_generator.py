import utils
import metafeatures_ssi
import metafeatures_landmarking
import os
import sys
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


if __name__ == "__main__":

    directory = 'C:\Users\silvia\Documents\silvia-data\\thesis_final\complete\\'

    dir_list = os.listdir(directory)

    try:
        output_file = sys.argv[1]
        if not output_file.endswith('.csv'):
            output_file += '.csv'

    except IndexError:
        output_file = 'metafeatures_complete.csv'

        print("Will save results to", output_file)

    with open(output_file, 'a') as fh:
        simple_names = ",".join(metafeatures_ssi.simple_metafeature_names())
        statistical_names = ",".join(metafeatures_ssi.statistical_metafeature_names())
        information_names = ",".join(metafeatures_ssi.information_theoretic_metafeature_names())
        time_names = ",".join(metafeatures_ssi.time_inf_names())
        landmarking_names = ",".join(metafeatures_landmarking.landmarking_metafeature_names())
        column_names = "{},{},{},{},{},{}\n".format("dataset", simple_names, statistical_names, information_names,
                                                    time_names, landmarking_names)
        fh.write(column_names)

    for e in dir_list:
        try:
            file_name = directory + os.path.basename(e)

            print os.path.basename(e)

            X, y, cat = utils.load_file(file_name)

            simple = metafeatures_ssi.simple_metafeatures(X, y, cat)
            stats = metafeatures_ssi.statistical_metafeatures(X, y, cat)
            info = metafeatures_ssi.information_theoretic_metafeatures(X, y, cat)
            time = metafeatures_ssi.time_metafeatures(simple["SimpleFeatureTime"], stats["StatisticalFeatureTime"],
                                                      info["ITFeatureTime"])
            print 'simple', simple
            print 'stats', stats
            print 'info', info
            print 'time', time

            X, y, cat = utils.load_file_landmarking(file_name)

            X = np.array(X)
            y = np.array(y)

            folds_split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
            folds = list(folds_split.split(X, y))

            land = metafeatures_landmarking.landmarker_metafeatures(X, y, folds)

            print 'landmarking', land

            with open(output_file, 'a') as fh:

                feature_list = [[os.path.basename(e)], simple.values(), stats.values(), info.values(), time.values(), land.values()]
                list_as_string = ",".join([str(item) for sublist in feature_list for item in sublist])
                fh.write(list_as_string + "\n")


        except AttributeError:
            pass


