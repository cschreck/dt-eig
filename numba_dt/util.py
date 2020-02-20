import numpy as np
from copy import deepcopy

from sklearn.impute import KNNImputer


def print_tree(tree):
    if tree.leaf:
        print('\t'*tree.depth, tree.class_distribution)
        return
    print('\t'*tree.depth, 'If', tree.split_column, '<=', tree.split_value)
    print_tree(tree.left_child)
    print('\t'*tree.depth, 'Else')
    print_tree(tree.right_child)


def get_depth(tree):
    depth = tree.depth
    if tree.leaf == True:
        return depth
    depth = max(get_depth(tree.left_child), depth)
    depth = max(get_depth(tree.right_child), depth)
    return depth


def get_size(tree):
    if tree.leaf == True:
        return 1
    else:
        return get_size(tree.left_child) + get_size(tree.right_child)


def make_missing_random(X, fixed=False):
    if fixed is False:
        feature_reliability = np.random.rand(X.shape[1])
    else:
        feature_reliability = np.ones(X.shape[1])*fixed
    value_survivability = np.random.rand(*X.shape)
    X_missing_random = deepcopy(X)
    X_missing_random = np.float64(X_missing_random)
    X_missing_random[value_survivability < feature_reliability] = np.nan
    return X_missing_random


def get_knn_for_na(X_missing, n_neighbors=5):
    knnimp = KNNImputer(n_neighbors=1)
    X_imputed = knnimp.fit_transform(X_missing)

    vals = {}
    for tup in np.argwhere(np.isnan(X_missing)):
        vals[tuple(tup)] = [X_imputed[tup[0], tup[1]]]
    for i in range(2, n_neighbors + 1):
        knnimp = KNNImputer(n_neighbors=i)
        X_imputed = knnimp.fit_transform(X_missing)

        for tup in np.argwhere(np.isnan(X_missing)):
            existing_vals = vals[tuple(tup)]
            new_mean = X_imputed[tup[0], tup[1]] * i
            for value in existing_vals:
                new_mean -= value

            existing_vals.append(new_mean)
            vals[tuple(tup)] = existing_vals

    X_imputed = np.ones(X_missing.shape) * np.nan
    indexed_vals = np.empty((len(vals), n_neighbors))
    for i, key in enumerate(vals):
        for j in range(n_neighbors):
            indexed_vals[i][j] = np.array(vals[key][j])
        X_imputed[key] = i

    return X_imputed, indexed_vals