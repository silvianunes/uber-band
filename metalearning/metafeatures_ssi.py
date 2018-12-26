from __future__ import division
from collections import OrderedDict
import numpy as np
import scipy.stats
import sklearn.metrics
import utils
import pandas as pd
import collections

simpleFeatureTime = 0
statisticalFeatureTime = 0
ITFeatureTime = 0
totalFeatureTime = 0



def simple_metafeature_names():
	return ["#Instances", "Log#Instances", "#Features", "Log#Features", "Dimensionality", "LogDimensionality",
            "InverseDimensionality", "LogInverseDimensionality", "#Classes", "#CategoricalFeatures",
           "%CategoricalFeatures", "#NumericFeatures", "%NumericFeatures", "RatioNumericalToNominal",
            "RatioNominalToNumerical", "ClassProbabilityMin",
			"ClassProbabilityMax", "ClassProbabilityMean", "ClassProbabilitySTD", "#BinaryFeatures", "SymbolsMin",
            "SymbolsMax", "SymbolsMean", "SymbolsSTD", "SymbolsSum", "#MissingValues", "%MissingValues",
            "#InstancesWithMissingValues", "%InstancesWithMissingValues", "#FeaturesWithMissingValues",
            "%FeaturesWithMissingValues", "MajorityClassSize", "MajorityClassPercentage",
            "SimpleFeatureTime"]


def simple_metafeatures(X, y, categorical):
    utils.input_check(X, y)
    features = OrderedDict()
    n = X.shape[0]
    p = X.shape[1]


    with utils.stopwatch() as sw:
        features["#Instances"] = n
        features["Log#Instances"] = np.log(n)
        features["#Features"] = p + 1
        features["Log#Features"] = np.log(p + 1)
        features["Dimensionality"] = (p + 1) / n
        features["LogDimensionality"] = np.log((p + 1)/n)
        features["InverseDimensionality"] = n / (p + 1)
        features["LogInverseDimensionality"] = np.log(n / (p + 1))

        classes, counts = np.unique(y, return_counts=True)

        num_var, cat_var = utils.data_types(X, categorical)

        nrNumeric = len(num_var)
        nrNominal = len(cat_var)

        features["#Classes"] = classes.shape[0]
        features["#CategoricalFeatures"] = nrNominal
        features["%CategoricalFeatures"] = (nrNominal * 100)/p
        features["#NumericFeatures"] = nrNumeric
        features["%NumericFeatures"] = (nrNumeric * 100)/p

        features["RatioNumericalToNominal"] = nrNumeric / nrNominal if nrNominal > 0 else 0
        features["RatioNominalToNumerical"] = nrNominal / nrNumeric if nrNumeric > 0 else 0

        class_probabilities = [count / n for count in counts]
        features["ClassProbabilityMin"] = np.min(class_probabilities)
        features["ClassProbabilityMax"] = np.max(class_probabilities)
        features["ClassProbabilityMean"] = np.mean(class_probabilities)
        features["ClassProbabilitySTD"] = np.std(class_probabilities)

        # symbols_per_column = [np.unique(column).shape[0] for column in X[:, np.where(cat_index)].T]

        symbols_per_column = [np.unique(X.loc[:, c]).shape[0] for c in X.columns if c in cat_var]

        if len(symbols_per_column) > 0:
            counter = collections.Counter(symbols_per_column)
            if counter.get(2L) is None:
                features["#BinaryFeatures"] = 0
            else:
                features["#BinaryFeatures"] = counter.get(2L)
            features["SymbolsMin"] = np.min(symbols_per_column)
            features["SymbolsMax"] = np.max(symbols_per_column)
            features["SymbolsMean"] = np.mean(symbols_per_column)
            features["SymbolsSTD"] = np.std(symbols_per_column)
            features["SymbolsSum"] = np.sum(symbols_per_column)
        else:
            features["BinaryFeatures"] = features["SymbolsMin"] = features["SymbolsMax"] = features["SymbolsMean"] = features["SymbolsSTD"] = features["SymbolsSum"] = 0

        missingValues = X.isnull().sum().sum()
        missingInstances = X.isnull().T.any().T.sum()
        missingFeatures = X.isnull().any().sum()
        features["#MissingValues"] = missingValues
        features["%MissingValues"] = (missingValues * 100)/(n * (p + 1))
        features["#InstancesWithMissingValues"] = missingInstances
        features["%InstancesWithMissingValues"] = (missingInstances * 100)/n
        features["#FeaturesWithMissingValues"] = missingFeatures
        features["%FeaturesInstancesWithMissingValues"] = (missingFeatures * 100)/n

        features["MajorityClassSize"] = max(counts)
        features["MajorityClassPercentage"] = (max(counts) * 100) / n


    features["SimpleFeatureTime"] = sw.duration
    # Missing value features missing for now since only datasets without missing features were selected.

    return features


def statistical_metafeature_names():
    return ["KurtosisMin", "KurtosisMax", "KurtosisMean", "KurtosisSTD",
			"SkewnessMin", "SkewnessMax", "SkewnessMean", "SkewnessSTD",
			"MeanSTDOfNumerical", "STDSTDOfNumerical", "MeanMeansOfNumerical",
            "StatisticalFeatureTime"]


def statistical_metafeatures(X, y, categorical):
    utils.input_check(X, y)
    features = OrderedDict()

    num_var, cat_var = utils.data_types(X, categorical)

    numerical = [X.columns.get_loc(c) for c in X.columns if c in num_var]

    # Statistical meta-features are only for the numerical attributes, if there are none, we list them as -1
    # we should see if there is a better way to deal with this, as -1 is a valid value for some of these features..
    if(sum(numerical) == 0):
        return OrderedDict.fromkeys(statistical_metafeature_names(), value=-1)

    with utils.stopwatch() as sw:
        # Taking taking kurtosis of kurtosis and skewness of kurtosis is suggested by Reif et al. in Meta2-features (2012)
        # kurtosisses = [scipy.stats.kurtosis(column[0]) for column in X[:,np.where(numerical)].T]

        kurtosisses = [scipy.stats.kurtosis(X.loc[:, c]) for c in X.columns if c in num_var]
        features["KurtosisMin"] = np.min(kurtosisses)
        features["KurtosisMax"] = np.max(kurtosisses)
        features["KurtosisMean"] = np.mean(kurtosisses)
        features["KurtosisSTD"] = np.std(kurtosisses)

        skewnesses = [scipy.stats.skew(X.loc[:, c]) for c in X.columns if c in num_var]
        features["SkewnessMin"] = np.min(skewnesses)
        features["SkewnessMax"] = np.max(skewnesses)
        features["SkewnessMean"] = np.mean(skewnesses)
        features["SkewnessSTD"] = np.std(skewnesses)

        standard_deviations = [np.std(X.loc[:, c]) for c in X.columns if c in num_var]
        features["MeanSTDOfNumerical"] = np.mean(standard_deviations)
        features["STDSTDOfNumerical"] = np.std(standard_deviations)

        means = [np.mean(X.loc[:, c]) for c in X.columns if c in num_var]
        features["MeanMeansOfNumerical"] = np.mean(means)

    features["StatisticalFeatureTime"] = sw.duration

    return features


def information_theoretic_metafeature_names():
    return ["ClassEntropy", "MeanFeatureEntropy", "MeanMutualInformation", "EquivalentNumberOfAttributes",
            "NoiseToSignalRatio", "ITFeatureTime"]


def information_theoretic_metafeatures(X, y, categorical):
    utils.input_check(X, y)

    X = X.fillna(0)

    features = OrderedDict()

    num_var, cat_var = utils.data_types(X, categorical)

    cat = [X.columns.get_loc(c) for c in X.columns if c in cat_var]

    classes, counts = np.unique(y, return_counts=True)

    class_entropy = scipy.stats.entropy(counts, base=2)
    features["ClassEntropy"] = class_entropy

    # Information theoretic meta-features below only apply to categorical values
    if(sum(cat) == 0):
        return OrderedDict.fromkeys(information_theoretic_metafeature_names(), value=-1)

    with utils.stopwatch() as sw:

        feature_entropies = [scipy.stats.entropy(X.loc[:, c]) for c in X.columns if c in cat_var]
        mean_feature_entropy = np.mean(feature_entropies)
        features["MeanFeatureEntropy"] = np.mean(mean_feature_entropy)

        mutual_informations = [sklearn.metrics.mutual_info_score(y, X.loc[:, c]) for c in X.columns if c in cat_var]
        mean_mutual_information = np.mean(mutual_informations)
        features["MeanMutualInformation"] = mean_mutual_information

        if mean_mutual_information == 0:
            features["EquivalentNumberOfAttributes"] = 0
        else:
            features["EquivalentNumberOfAttributes"] = class_entropy/mean_mutual_information

        if mean_mutual_information == 0:
            features["NoiseToSignalRatio"] = 0
        else:
            features["NoiseToSignalRatio"] = (mean_feature_entropy - mean_mutual_information) / mean_mutual_information

    features["ITFeatureTime"] = sw.duration

    return features


def time_inf_names():
    # return ["SimpleFeatureTime", "StatisticalFeatureTime", "ITFeatureTime", "TotalFeatureTime"]
    return ["TotalFeatureTime"]


def time_metafeatures(simple, stats, it):
    features = OrderedDict()
    if simple is -1:
        simple = 0
    if stats is -1:
        stats = 0
    if it is -1:
        it = 0

    # features["SimpleFeatureTime"] = simple
    # features["StatisticalFeatureTime"] = stats
    # features["ITFeatureTime"] = it
    features["TotalFeatureTime"] = sum([simple, stats, it])

    return features

if __name__ == "__main__":
    print 'success'
