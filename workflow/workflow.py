from common_defs import *
import importlib

from utils import weighted_sample


def get_workflow(space):
    params = weighted_sample(space)
    print params

    return params


def get_hyperparms(params):
    l = str("defs."+params['learner'])
    learner = importlib.import_module(l)

    pr_ln = learner.get_params()

    if params['featureSelection'] != None:
        f = str("defs.dimreduction.eval."+params['featureSelection'])
        fs = importlib.import_module(f)
        pr_fs = fs.get_params()
        params = (params, pr_ln, pr_fs)
    else:
        params = (params, pr_ln)

    print params
    return params


def try_params(n_instances, params, train, valid, test, istest):
    pprint(params)

    wf = params[0]

    l = str("defs." + wf['learner'])
    learner = importlib.import_module(l)
    ln = learner.get_class(params[1])

    if wf['featureSelection'] != None:
        f = str("defs.dimreduction.eval." + wf['featureSelection'])
        fs = importlib.import_module(f)
        wfw = fs.get_evaluator(params[2], ln)
    else:
        wfw = ln

    if istest:
        result = test_weka_classifier(wfw, train, test)
    else:
        result = train_and_eval_weka_classifier(wfw, train, valid, n_instances)

    return result

