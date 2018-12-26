import arff
import numpy as np
import os
import pandas as pd

directory = 'C:\Users\silvia\Desktop\\results new meta\\cfs\\'

def compile_(directory, extension='.csv'):
    dir_list = os.listdir(directory)
    for e in dir_list:
        if e.endswith(extension):
            try:
                fileName = directory + os.path.basename(e)

                print fileName

                dataset = np.genfromtxt(fileName, delimiter=',')

                dataset = dataset[1:, :]

                data = pd.read_csv(fileName)

                auc = 0

                count = 1

                # df = pd.DataFrame(columns=('dataset', 'auc'))

                rows_list = []

                for i in range(dataset.shape[0]):
                    auc += dataset[i, 41]
                    # print dataset[i, 1]
                    if np.mod(count, 10) == 0:
                        med_auc = auc/10
                        # print data.loc[i, "Key_Dataset"]
                        # print med_auc
                        rows_list.append((data.loc[i, "Key_Dataset "], med_auc))
                        auc = 0
                    count +=1

                print rows_list

                df = pd.DataFrame(rows_list)

                df.to_csv(os.path.basename(e)+'.csv', sep=',')

                # np.savetxt('C:\Users\silvia\Downloads\datasetsOpenML\csv\\'+os.path.basename(e), dat, fmt='%10.3f', delimiter=',')

            except MemoryError:
                pass

    return "success"

compile_(directory)