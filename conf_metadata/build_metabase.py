import arff
import numpy as np
import os
import pandas as pd

directory = 'C:\Users\silvia\Dropbox\experimentos  new - compilados\CFS\\'

maindata = pd.read_csv('C:\Users\silvia\Dropbox\experimentos  new - compilados\\base.csv')

def compile_(directory, extension='.csv'):
    dir_list = os.listdir(directory)
    for e in dir_list:
        if e.endswith(extension):
            try:
                fileName = directory + os.path.basename(e)

                print fileName

                dataset = pd.read_csv(fileName)
                rows_list = []

                for i in range(maindata.shape[0]):
                    value = None
                    for j in range(dataset.shape[0]):
                        if maindata.loc[i, 'Datasets'] == dataset.loc[j, '0']:
                            value = dataset.loc[j, '1']
                    rows_list.append(value)

                maindata[os.path.basename(e)] = pd.DataFrame(rows_list)

            except MemoryError:
                pass
    maindata.to_csv('CFS.csv', sep=',')

    return "success"

compile_(directory)