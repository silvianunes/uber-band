import pandas as pd
import numpy as np
from operator import itemgetter

import cPickle as pickle
import os

# # dir_data = 'C:\Users\silvia\Documents\Silvia-data\\auto-weka datasets\\done\\'
# # dir_results = 'C:\Users\silvia\Dropbox\\ab-cl\\test\\'
# #
# #
# # def main(directory_data, directory_results):
# #
# #     dir_d = os.listdir(directory_data)
# #     dir_r = os.listdir(directory_results)
# #     columns = ['data',  'err', 'time']
# #     columns2 = ['data', 'err', 'time', 'param']
# #     df_average = pd.DataFrame(columns=columns)
# #     df_complete = pd.DataFrame(columns=columns2)
# #
# #     for e in dir_d:
# #         print('NAME:    ', os.path.basename(e))
# #
# #         count = 0
# #         time = 0
# #         err = 0
# #
# #         for i in dir_r:
# #
# #             # print(os.path.basename(e), os.path.basename(i))
# #
# #             if os.path.basename(e) in os.path.basename(i):
# #
# #                 count += 1
# #
# #                 try:
# #                     file = directory_results + os.path.basename(i)
# #
# #                     with open(file, 'rb') as f:
# #                         data = pickle.load(f)
# #
# #                     # print('unsorted:', data)
# #
# #                     data = sorted(data, key=itemgetter('err'))
# #
# #                     print('sorted:', data)
# #
# #                     data = data[0]
# #                     time += data['seconds']
# #                     err += data['err']
# #                     param = data['params']
# #
# #
# #                     df_complete = df_complete.append({'data': [os.path.basename(i)], 'time': data['seconds'], "err": ([data['err']]), "param": param}, ignore_index=True)
# #
# #                 except ValueError:
# #                     pass
# #         if count!=0:
# #             time = time/count
# #             err = err/count
# #             df_average = df_average.append({'data': [os.path.basename(e)], 'time': [time], "err": [err]}, ignore_index=True)
# #
# #             print(df_average)
# #
# #     df_average.to_csv('average-results-autoband-test.csv', sep=',')
# #     df_complete.to_csv('complete-results-autoband-test.csv', sep=',')
# #
# #     return 'finished'
#
#
#
# main(dir_data, dir_results)


import pandas as pd
import numpy as np
from operator import itemgetter

import cPickle as pickle
import os

dir_data = 'C:\Users\silvia\Documents\silvia-data\\thesis_final\\benchmarkings\\'
dir_results = 'C:\Users\silvia\Dropbox\THESIS_EXP_FINAL\hiper-parametros\\nmax\\11\\valid\\'


def main(directory_data, directory_results):

    dir_d = os.listdir(directory_data)
    dir_r = os.listdir(directory_results)
    columns = ['data', 'loss_m', 'err_m', 'auc_m', 'time_m', 'loss_d', 'err_d', 'auc_d', 'time_d']
    columns2 = ['data', 'loss', 'err', 'auc', 'time', 'param']
    df_average = pd.DataFrame(columns=columns)
    df_complete = pd.DataFrame(columns=columns2)

    for e in dir_d:
        print('DATA:    ', os.path.basename(e))

        count = 0
        loss = []
        time = []
        auc = []
        err = []

        for i in dir_r:

            if os.path.basename(e) in os.path.basename(i):

                count += 1

                try:
                    file = directory_results + os.path.basename(i)

                    with open(file, 'rb') as f:
                        data = pickle.load(f)

                    # print 'unsorted:', data

                    data = sorted(data, key=itemgetter('auc'), reverse=True)

                    # print 'sorted',  data


                    first = data[0]
                    loss.append(first['loss'])
                    time.append(first['seconds'])
                    auc.append(first['auc'])
                    err.append(first['err'])
                    # param = first['params']


                    df_complete = df_complete.append({'data': [os.path.basename(i)], 'loss': [first['loss']], 'time': [first['seconds']], "auc": [first['auc']], "err": [first['err']], "param": [first['params']]}, ignore_index=True)

                except ValueError:
                    pass
        if count!=0:
            print auc

            loss_m = np.mean(loss)
            time_m = np.mean(time)
            auc_m = np.mean(auc)
            err_m = np.mean(err)

            loss_d = np.std(loss)
            time_d = np.std(time)
            auc_d = np.std(auc)
            err_d = np.std(err)

            df_average = df_average.append({'data': [os.path.basename(e)], 'loss_m': [loss_m], 'time_m': [time_m], "auc_m":
                [auc_m], "err_m": [err_m], 'loss_d': [loss_d], 'time_d': [time_d], "auc_d": [auc_d], "err_d": [err_d]}, ignore_index=True)

            print(df_average)

    df_average.to_csv('average-results-phb_meta_test.csv', sep=',')
    df_complete.to_csv('complete-results-phb_meta_test.csv', sep=',')

    return 'finished'


def save_all(directory_results):
    dir_r = os.listdir(directory_results)
    columns2 = ['data', 'counter', 'loss', 'err', 'auc', 'time', 'param']
    df_complete = pd.DataFrame(columns=columns2)

    for e in dir_r:
        print 'DATA:    ', os.path.basename(e)
        df_complete = pd.DataFrame(columns=columns2)

        try:
            file = directory_results + os.path.basename(e)

            with open(file, 'rb') as f:
                data = pickle.load(f)

            # print 'unsorted:', data

            data = sorted(data, key=itemgetter('auc'), reverse=True)

            # print 'sorted',  data

            for d in data:
                df_complete = df_complete.append(
                    {'data': [os.path.basename(e)], 'counter': [d['counter']], 'loss': [d['loss']],
                     'time': [d['seconds']],
                     "auc": [d['auc']], "err": [d['err']], "param": [d['params']]},
                    ignore_index=True)

            df_complete.to_csv('valid_'+os.path.basename(e)+'.csv', sep=',')

        except ValueError:
            pass

    return 'finished'


def save_all_valid(directory_data, directory_results):
    dir_d = os.listdir(directory_data)
    dir_r = os.listdir(directory_results)

    for e in dir_d:
        print('DATA:    ', os.path.basename(e))
        df_complete = pd.DataFrame()

        for i in dir_r:

            if os.path.basename(e) in os.path.basename(i):

                try:
                    file = directory_results + os.path.basename(i)

                    with open(file, 'rb') as f:
                        data = pickle.load(f)

                    data = sorted(data, key=itemgetter('counter'))

                    auc = []
                    for d in data:
                        print d
                        auc.append(d['auc'])

                    df_complete["auc_"+os.path.basename(i)] = pd.Series(auc)

                except ValueError:
                    pass
            print df_complete

        df_complete['average'] = df_complete.mean(numeric_only=True, axis=1)
        df_complete.to_csv('meta_' + os.path.basename(e) + '.csv', sep=',')

    return 'finished'

save_all_valid(dir_data, dir_results)




