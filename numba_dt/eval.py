import numpy as np
from numba import njit


@njit(cache=True)
def information_gain_nan(X, y, weights, split_attribute):
    nan_idx = np.isnan(X[:, split_attribute])
    notnan_idx = np.invert(nan_idx)
    total_weight = np.sum(weights)
    weight_nan = np.sum(weights[nan_idx]) / total_weight
    weight_notnan = np.sum(weights[notnan_idx]) / total_weight
    split_entropy_nan = entropy(y[nan_idx], weights[nan_idx]) * weight_nan
    split_entropy_notnan = entropy(y[notnan_idx],
                                   weights[notnan_idx]) * weight_notnan
    split_entropy = split_entropy_nan + split_entropy_notnan
    return entropy(y, weights) - split_entropy


@njit(cache=True)
def c45_information_gain(X, y, weights, split_value, split_attribute, alpha):
    #beta = np.sum(np.isnan(X[:, split_attribute])) / X.shape[0]
    if alpha == -1:
        beta = np.sum(weights[np.isnan(X[:, split_attribute])]) / np.sum(weights)
        alpha = 1 - beta

    missing_idx = np.isnan(X[:, split_attribute])
    non_missing_idx = np.invert(missing_idx)

    y_non_missing = y[non_missing_idx]
    weights_non_missing = weights[non_missing_idx]
    X_non_missing = X[non_missing_idx]

    ig = alpha * (entropy(y, weights) - split_entropy_score(
        X_non_missing,
        y_non_missing,
        weights_non_missing,
        split_value,
        split_attribute))
    return alpha * ig

@njit(cache=True)
def hexpected_information_gain(X, X_imputed, y, weights, split_value,
                               split_column, alpha):
    if alpha == -1:
        beta = np.sum(weights[np.isnan(X[:, split_column])]) / np.sum(weights)
        alpha = 1 - beta
    else:
        beta = 1-alpha
    if alpha == 1:
        return information_gain(X, y, weights, split_value, split_column)

    missing_idx = np.isnan(X[:, split_column])
    non_missing_idx = np.invert(missing_idx)
    y_non_missing = y[non_missing_idx]
    weights_non_missing = weights[non_missing_idx]
    X_non_missing = X[non_missing_idx]

    igimputed = information_gain(X_imputed[missing_idx], y[missing_idx],
                                 weights[missing_idx], split_value,
                                 split_column)

    ig = information_gain(X_non_missing, y_non_missing, weights_non_missing,
                          split_value, split_column)
    return alpha * ig + beta * igimputed

@njit(cache=True)
def entropy(y, weights):
    weight_sum = np.sum(weights)
    classes = np.unique(y)
    class_weights = [np.sum(weights[y == clas])/weight_sum for clas in
                     classes]
    return -np.sum(np.array([x * np.log2(x) for x in class_weights]))

@njit(cache=True)
def information_gain(X, y, weights, split_value, split_attribute):
    split_entropy = split_entropy_score(X, y, weights, split_value,
                                        split_attribute)
    return entropy(y, weights) - split_entropy

@njit(cache=True)
def split_entropy_score(X, y, weights, split_value, split_attribute):
    left_idx = X[:, split_attribute] <= split_value
    right_idx = np.invert(left_idx)
    total_weight = np.sum(weights)
    weight_left = np.sum(weights[left_idx]) / total_weight
    weight_right = np.sum(weights[right_idx]) / total_weight
    split_entropy_left = entropy(y[left_idx], weights[left_idx]) * weight_left
    split_entropy_right = entropy(y[right_idx],
                                  weights[right_idx]) * weight_right
    split_entropy = split_entropy_left + split_entropy_right
    return split_entropy

