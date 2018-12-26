import weka.core.jvm as jvm

jvm.start()

import sys
import cPickle as pickle
from pprint import pprint

from hyperband import Hyperband

import os

# loading data
from load_data import Load_Data

from workflow.workflow import get_hyperparms, try_params, get_workflow

from metalearning.weights_generator import weights_generator
from workflow.utils import get_space


from time import time, ctime

dir = 'C:\Users\silvia\Documents\silvia-data\\thesis_final\\test_HP\\'
# dir_done = 'C:\Users\silvia\Documents\silvia-data\\thesis_final\done\\'


def main(directory):

    dir_list = os.listdir(directory)
    for e in dir_list:
        file_name = directory + os.path.basename(e)

        load = Load_Data(file_name)

        train, valid, test = load.split_train_test_valid()

        # get workflow weights
        wf_weights, time_ml = weights_generator(file_name, os.path.basename(e))

        # get search space

        space = get_space(wf_weights)

        # print space

        for i in range(5):
            try:
                output_file1 = sys.argv[1]
                output_file2 = sys.argv[1]
                if not output_file1.endswith('.pkl'):
                    output_file1 += '.pkl'
                    output_file2 += '.pkl'
            except IndexError:
                # output_file1 = 'results_ub_' + os.path.basename(e) + '_' + str(i) + '.pkl'
                # output_file2 = 'results_ub_test_' + os.path.basename(e) + '_' + str(i) + '.pkl'
                output_file1 = 'ub_eta2_' + os.path.basename(e) + '_' + str(i) + '.pkl'
                output_file2 = 'ub_eta2_test_' + os.path.basename(e) + '_' + str(i) + '.pkl'

                print("Will save results to", output_file1, output_file2)

            # data = load(file_name)

            start_time_validation = time()

            hb = Hyperband(get_workflow, get_hyperparms, try_params, train, valid, test, space)
            results, all_results = hb.run(skip_last=1)

            finish_time_validation = int(round(time() - start_time_validation)) + time_ml

            start_time_test = time()

            test_results = hb.tests(results)

            finish_time_test = int(round(time() - start_time_test))

            print("{} total, best in validation:\n".format(len(results)))

            for r in sorted(all_results, key=lambda x: x['auc'], reverse=True):
                print("auc: {:.2} | {} seconds | {:.1f} instances | run {} ".format(
                           r['auc'], r['seconds'], r['instances'], r['counter']))
                pprint(r['params_complete'])
                print

            all_results.append(finish_time_validation)

            print("test results")
            for r in range(len(test_results)):
                t = test_results[r]
                print("loss: {:.2%} | auc: {:.2%} | {} seconds | {} run ".format(
                    t['loss'], t['auc'], t['seconds'], t['counter']))
                pprint(t['params'])
                print

            test_results.append(finish_time_test)

            print("results:     ", all_results)
            print("test results:    ", test_results)
            print("saving...")

            with open(output_file1, 'wb') as f:
                       pickle.dump(results, f)

            with open(output_file2, 'wb') as f:
                       pickle.dump(test_results, f)

    return 'finished'


main(dir)


jvm.stop()