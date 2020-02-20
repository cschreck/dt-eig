from copy import deepcopy
import numpy as np
from .eval import hexpected_information_gain
from numba import njit
from numba import prange


class EIGDecisionTree:
    def __init__(self, max_depth=20, criterion='heig',
                 conditional_reliability=False):
        self.max_depth = max_depth
        self.general_params = dict()
        self.general_params['criterion'] = criterion
        self.general_params['cond_reliability'] = conditional_reliability

    def fit(self, X, X_imputed, y, weights=None, neighbors=None):
        if weights is None:
            weights = np.ones(y.shape)

        class_labels = np.unique(y)
        self.general_params['class_labels'] = class_labels
        self.general_params['neighbors'] = neighbors

        reliabilities = np.sum(np.isnan(X), axis=0)/X.shape[0]
        self.general_params['reliability'] = 1 - reliabilities

        self.build_tree(X, X_imputed, y, weights, self.general_params)

    def build_tree(self, X, X_imputed, y, weights, general_params):
        self.tree = HNode(0, general_params)

        if X_imputed is None:
            X_imputed = self.impute(X)

        self.tree.split(X, X_imputed, y, weights, self.max_depth)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        if np.sum(np.isnan(X)) > 0:
            return np.array([self._predict_row_with_missing(x) for x in X])
        return np.array([self._predict_row(x) for x in X])

    def _predict_row(self, x):
        node = self.tree
        while node.leaf is False:
            if x[node.split_column] <= node.split_value:
                node = node.left_child
            else:
                node = node.right_child

        return node.class_distribution

    def _predict_row_with_missing(self, x):
        node = self.tree
        dist, _ = _predict_rec(x, node)
        return dist


    def impute(self, X_missing):
        pass


def _predict_rec(x, node):
    if node.leaf:
        return node.class_distribution, node.node_weight
    if np.isnan(x[node.split_column]):
        a = _predict_rec(x, node.left_child)
        b = _predict_rec(x, node.right_child)
        dist_l, weight_l = a[0], a[1]
        dist_r, weight_r = b[0], b[1]
        total_weight = (weight_l + weight_r)
        return (dist_l*weight_l+dist_r*weight_r)/total_weight, total_weight
    elif x[node.split_column] <= node.split_value:
        return _predict_rec(x, node.left_child)
    else:
        return _predict_rec(x, node.right_child)


class HNode:

    def __init__(self, depth, general_params):
        self.depth = depth
        self.left_child = None
        self.right_child = None
        self.split_column = None
        self.split_value = None
        self.leaf = False
        self.node_weight = -1
        self.general_params = general_params

    def split(self, X, X_imputed, y, weights, max_depth):
        if self.depth == max_depth \
                or X.shape[0] < 2 \
                or np.sum(weights) < 2:
            self.node_weight = np.sum(weights)
            self.leaf = True
            self.class_distribution = self.get_class_distribution(y, weights)
            return self

        self.split_column, self.split_value = self.find_best_split(X, X_imputed,
                                                                   y,
                                                                   weights)

        if self.split_column is None:
            self.leaf = True
            self.node_weight = np.sum(weights)
            self.class_distribution = self.get_class_distribution(y, weights)
            return self
        else:
            # update weights
            nan_idx = np.isnan(X[:, self.split_column])
            left_idx = X[:, self.split_column] <= self.split_value
            left_idx = np.logical_or(left_idx, nan_idx)
            right_idx = X[:, self.split_column] > self.split_value
            right_idx = np.logical_or(right_idx, nan_idx)

            sum_weight_not_nan = np.sum(weights[left_idx]) + np.sum(
                weights[right_idx])
            left_nan_weight = np.sum(weights[left_idx]) / sum_weight_not_nan
            right_nan_weight = np.sum(weights[right_idx]) / sum_weight_not_nan
            left_weights = deepcopy(weights)
            right_weights = deepcopy(weights)
            for index in np.argwhere(
                    np.isnan(X[:, self.split_column])).flatten():
                left_weights[index] = weights[index] * left_nan_weight
                right_weights[index] = weights[index] * right_nan_weight
            #
            self.left_child = HNode(
                self.depth + 1,
                self.general_params
            ).split(
                X[left_idx],
                X_imputed[left_idx],
                y[left_idx],
                left_weights[left_idx],
                max_depth
            )
            self.right_child = HNode(
                self.depth + 1,
                self.general_params
            ).split(
                X[right_idx],
                X_imputed[right_idx],
                y[right_idx],
                right_weights[right_idx],
                max_depth
            )

        return self

    def find_best_split(self, X, X_imputed, y, weights):
        best_column = None
        best_split_value = None
        best_ig = 0


        c_v_pairs = []
        for column in range(X.shape[1]):
            values = np.unique(X[:, column])
            values = values[np.invert(np.isnan(values))]
            values = [(a + b) / 2 for a, b in zip(values[:-1], values[1:])]
            for value in values:
                c_v_pairs.append((column, value))

        c_v_pairs = np.array(c_v_pairs)

        if not self.general_params['cond_reliability']:
            reliabilities = self.general_params['reliability']
        else:
            reliabilities = np.ones(X.shape[1]) * - 1
        if len(c_v_pairs) > 0:
            if self.general_params['criterion'] == 'heig':
                res = get_ig_for_cvp(X,
                                     X_imputed,
                                     y,
                                     weights,
                                     c_v_pairs,
                                     reliabilities
                                     )
        else:
            return best_column, best_split_value

        #no gain
        if np.max(res) < 0.00000001:
            return best_column, best_split_value

        best_column, best_split_value = c_v_pairs[np.argmax(res)]
        return np.int(best_column), best_split_value

    def get_class_distribution(self, y, weights):
        class_distribution = []
        for label in self.general_params['class_labels']:
            temp = np.zeros(y.shape[0])
            temp[y == label] = 1
            class_distribution.append(np.sum(temp * weights) / np.sum(weights))

        return np.array(class_distribution)


@njit(cache=True, parallel=True)
def get_ig_for_cvp(X, X_imputed, y, weights, cvp, reliabilities):
    res = np.empty(cvp.shape[0])

    for i in prange(cvp.shape[0]):
        pair = cvp[i]
        column = np.int(pair[0])
        value = np.float(pair[1])
        alpha = reliabilities[column]
        result = hexpected_information_gain(X, X_imputed, y, weights, value,
                                            column, alpha)
        res[i] = np.float64(result)

    return res

