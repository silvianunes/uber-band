from weka.attribute_selection import ASEvaluation
from weka.classifiers import Classifier

import defs.dimreduction.search.rk as rk
from common_defs import *


space = {
    'binarizeNumericAttributes': hp.choice('binarizeNumericAttributes', (True, False)),
    'missingMerge': hp.choice('missingMerge', (True, False)),
}

def get_params():
    params = sample(space)
    return handle_integers(params)
#


def get_evaluator(params, base):
    pprint(params)

    L = list([])

    if params['missingMerge'] == False:
        L.append("-M")

    if params['binarizeNumericAttributes'] == True:
        L.append("-B")

    param_search = rk.get_params()

    search = rk.get_search(param_search)

    # search = ASSearch(classname="weka.attributeSelection.Ranker")
    evaluator = ASEvaluation(classname="weka.attributeSelection.InfoGainAttributeEval", options=L)

    clf = Classifier(classname="weka.classifiers.meta.AttributeSelectedClassifier")

    clf.set_property("evaluator", evaluator.jobject)
    clf.set_property("search", search.jobject)
    clf.set_property("base", base.jobject)

    return clf


def try_params(n_instances, params, base, train, valid, test, istest):
    n_instances = int(round(n_instances))
    # print "n_instances:", n_instances
    pprint(params)

    L = list([])

    if params['missingMerge'] == False:
        L.append("-M")

    if params['binarizeNumericAttributes'] == True:
        L.append("-B")

    param_search = rk.get_params()

    search = rk.get_class(param_search)

    # search = ASSearch(classname="weka.attributeSelection.Ranker")
    evaluator = ASEvaluation(classname="weka.attributeSelection.InfoGainAttributeEval", options=L)

    clf = Classifier(classname="weka.classifiers.meta.AttributeSelectedClassifier")

    clf.set_property("evaluator", evaluator.jobject)
    clf.set_property("search", search.jobject)
    clf.set_property("base", base.jobject)

    if istest:
        result = test_weka_classifier(clf, train, test)
    else:
        result = train_and_eval_weka_classifier(clf, train, valid, n_instances)

    return result