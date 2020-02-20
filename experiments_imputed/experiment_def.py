import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from numba_dt.htree import EIGDecisionTree
from numba_dt.ctree import C45DecisionTree
from numba_dt.util import make_missing_random
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer


def experiment_setting_1(X, y, runs=5, missingness=0.1):
    results = []
    for i in range(runs):
        np.random.seed(i)
        X_missing = make_missing_random(X, missingness)

        ss = StratifiedKFold(shuffle=True, random_state=i)

        for train_index, test_index in ss.split(X, y):
            X_train = X_missing[train_index]
            y_train = y[train_index]
            imputer = KNNImputer()
            imputer.fit(X_train)
            X_test = imputer.transform(X_missing[test_index])
            y_test = y[test_index]

            X_train_imputed = np.ones(X_train.shape) * np.nan
            for idx in np.argwhere(np.isnan(X_missing[train_index])):
                X_train_imputed[idx[0], idx[1]] = X_train[idx[0], idx[1]]

            hdt = EIGDecisionTree(max_depth=20)
            hdt.fit(X_train, X_train_imputed, y_train)
            results.append(accuracy_score(hdt.predict(X_test), y_test))
            #print(get_depth(hdt.tree), get_size(hdt.tree))

    return results


def experiment_setting_2(X, y, runs=5, missingness=0.1):
    results = []
    for i in range(runs):
        np.random.seed(i)
        X_missing = make_missing_random(X, missingness)

        ss = StratifiedKFold(shuffle=True, random_state=i)

        for train_index, test_index in ss.split(X, y):
            X_train = X_missing[train_index]
            y_train = y[train_index]
            imputer = KNNImputer()
            imputer.fit(X_train)
            X_test = imputer.transform(X_missing[test_index])
            y_test = y[test_index]

            knnimp = KNNImputer()
            X_knn_full_imputed = knnimp.fit_transform(X_train)
            X_train_imputed = np.ones(X_train.shape) * np.nan
            for idx in np.argwhere(np.isnan(X_train)):
                X_train_imputed[idx[0], idx[1]] = X_knn_full_imputed[
                    idx[0], idx[1]]

            hdt = EIGDecisionTree(max_depth=20)
            hdt.fit(X_train, X_train_imputed, y_train)
            results.append(accuracy_score(hdt.predict(X_test), y_test))
            #print(get_depth(hdt.tree), get_size(hdt.tree))

    return results


def experiment_setting_3(X, y, runs=5, missingness=0.1):
    results = []
    for i in range(runs):
        np.random.seed(i)
        X_missing = make_missing_random(X, missingness)

        ss = StratifiedKFold(shuffle=True, random_state=i)

        for train_index, test_index in ss.split(X, y):
            X_train = X_missing[train_index]
            y_train = y[train_index]
            imputer = KNNImputer()
            imputer.fit(X_train)
            X_test = imputer.transform(X_missing[test_index])
            y_test = y[test_index]

            dt = C45DecisionTree(criterion='c45', max_depth=20)
            dt.fit(X_train, y_train)
            results.append(accuracy_score(dt.predict(X_test), y_test))
            #print(get_depth(dt.tree), get_size(dt.tree))

    return results


def experiment_setting_4(X, y, runs=5, missingness=0.1):
    results = []
    for i in range(runs):
        np.random.seed(i)
        X_missing = make_missing_random(X, missingness)

        ss = StratifiedKFold(shuffle=True, random_state=i)

        for train_index, test_index in ss.split(X, y):
            X_train = X_missing[train_index]
            y_train = y[train_index]
            imputer = KNNImputer()
            imputer.fit(X_train)
            X_test = imputer.transform(X_missing[test_index])
            y_test = y[test_index]

            si = SimpleImputer()
            X_train = si.fit_transform(X_train)

            dt = C45DecisionTree(criterion='c45', max_depth=20)
            dt.fit(X_train, y_train)
            results.append(accuracy_score(dt.predict(X_test), y_test))

    return results


def experiment_setting_5(X, y, runs=5, missingness=0.1):
    results = []
    for i in range(runs):
        np.random.seed(i)
        X_missing = make_missing_random(X, missingness)

        ss = StratifiedKFold(shuffle=True, random_state=i)

        for train_index, test_index in ss.split(X, y):
            X_train = X_missing[train_index]
            y_train = y[train_index]
            imputer = KNNImputer()
            imputer.fit(X_train)
            X_test = imputer.transform(X_missing[test_index])
            y_test = y[test_index]

            si = KNNImputer()
            X_train = si.fit_transform(X_train)

            dt = C45DecisionTree(criterion='c45', max_depth=20)
            dt.fit(X_train, y_train)
            results.append(accuracy_score(dt.predict(X_test), y_test))

    return results
