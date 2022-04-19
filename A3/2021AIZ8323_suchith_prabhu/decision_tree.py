import os
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from pandas.api.types import is_string_dtype, is_categorical_dtype, is_numeric_dtype

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    prob = counts/counts.sum()
    return -1*np.sum(prob*np.log(prob))

def train_category(df):
    for n, c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()

def apply_category(df, trn):
    for n,c in df.items():
        if (n in trn.columns) and (trn[n].dtype.name=='category'):
            df[n] = c.astype('category').cat.as_ordered()
            df[n] = df[n].cat.set_categories(trn[n].cat.categories, ordered=True)

def convert_numerical(df, min_n_ord=0):
    is_cat_col = []
    for n, c in df.items():
        if is_categorical_dtype(c) and len(df[n].cat.categories) > min_n_ord:
            df[n] = c.cat.codes
            is_cat_col.append(True)
        elif is_categorical_dtype(c):
            is_cat_col.extend([True]*len(df[n].cat.categories))
        else:
            is_cat_col.append(False)
    return is_cat_col


def proc_df(df, y_label, min_n_ord=0):
    X, y = None, None
    df = df.copy()

    y = df[y_label]
    if is_categorical_dtype(y): y = (y.cat.codes).values
    df.drop(y_label, axis=1, inplace=True)

    is_cat_col = convert_numerical(df, min_n_ord)
    df = pd.get_dummies(df)
    X = df.values
    return X, y, is_cat_col

def proc_onehot(dfs, y_label):
    X_df = pd.concat(dfs)
    y_df = X_df[y_label]
    X_df.drop(y_label, axis=1, inplace=True)

    X_oh = pd.get_dummies(X_df, columns=X_df.columns, prefix=X_df.columns)

    X, y = [], []
    start_idx = 0
    for df in dfs:
        end_idx = start_idx+df.shape[0]
        X.append(X_oh[start_idx:end_idx].values)
        if is_categorical_dtype(y_df):
            y.append((y_df[start_idx:end_idx].cat.codes).values)
        else:
            y.append(y_df[start_idx:end_idx].values)
        start_idx = end_idx

    return X, y


class DecisionTree():
    def __init__(self, x, y, is_cat_col=None, idxs=None, min_leaf=4,
                 depth=0, max_depth=None):
        self.x, self.y, self.idxs, self.is_cat_col, self.min_leaf = x, y, idxs, is_cat_col, min_leaf
        if idxs is None: self.idxs = np.arange(len(y))

        self.n, self.c = len(self.idxs), x.shape[1]
        self.score = float('inf')
        self.children = []
        self.depth = depth
        self.max_depth = max_depth

        labels, label_count = np.unique(self.y[self.idxs], return_counts=True)
        self.val = labels[np.argmax(label_count)]

        self.split_var, self.split_val, self.split_col = None, None, None

        self.find_varsplit()


    def find_varsplit(self):
        if self.max_depth is not None and self.depth >= self.max_depth:
            return

        for i in range(self.c):
            if self.is_cat_col is not None and self.is_cat_col[i]:
                self.find_split_category(i)
            else:
                self.find_split_numerical(i)

        if self.is_leaf:
            return

        x = self.split_col
        if self.is_cat_col is not None and self.is_cat_col[self.split_var]:
            for attr in sorted(self.split_val):
                attr_mask = np.nonzero(x == attr)[0]
                self.children.append(DecisionTree(self.x, self.y, self.is_cat_col,
                                                  self.idxs[attr_mask], self.min_leaf,
                                                  max_depth=self.max_depth,
                                                  depth=self.depth+1))
        else:
            lr_masks = []
            lr_masks.append( np.nonzero(x <= self.split_val)[0] )
            lr_masks.append( np.nonzero(x > self.split_val)[0] )
            for mask in lr_masks:
                self.children.append(DecisionTree(self.x, self.y, self.is_cat_col,
                                                  self.idxs[mask], self.min_leaf,
                                                  max_depth=self.max_depth,
                                                  depth=self.depth+1))


    def find_split_category(self, var_idx):
        x, y = self.x[self.idxs, var_idx], self.y[self.idxs]
        if len(y) < self.min_leaf or len(np.unique(y)) == 1:
            return

        idx_sort = np.argsort(x)
        sorted_x = x[idx_sort]
        attrs, idx_start, attrs_cnt = np.unique(sorted_x, return_index=True, return_counts=True)
        attrs_idx = np.split(idx_sort, idx_start[1:])

        if len(attrs) == 1:
            return

        curr_score = 0
        attrs_dict = {}
        for i in range(len(attrs)):
            attr_prob = attrs_cnt[i]/self.n
            attr_entropy = entropy(y[attrs_idx[i]])
            curr_score += attr_prob*attr_entropy
            attrs_dict[attrs[i]] = i

        if curr_score < self.score:
            self.score, self.split_var = curr_score, var_idx
            self.split_val = attrs_dict
            self.split_col = x


    def find_split_numerical(self, var_idx):
        x, y = self.x[self.idxs, var_idx], self.y[self.idxs]
        if len(y) < self.min_leaf or len(np.unique(y)) == 1:
            return

        median = np.median(x)

        lr_mask = []
        lr_mask.append(x <= median)
        lr_mask.append(x > median)

        curr_score = 0
        for mask in lr_mask:
            count = mask.sum()
            if count < self.min_leaf:
                return
            entp = entropy(y[mask])
            prob = count/self.n
            curr_score += prob*entp

        if curr_score < self.score:
            self.score, self.split_var = curr_score, var_idx
            self.split_val = median
            self.split_col = x


    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf: return self.val
        if self.is_cat_col is not None and self.is_cat_col[self.split_var]:
            child_idx = self.split_val.get( xi[self.split_var],
                                           np.random.randint(len(self.split_val)) )
            t = self.children[child_idx]
        else:
            t = self.children[0] if xi[self.split_var] <= self.split_val else self.children[1]
        return t.predict_row(xi)

    def accuracy(self, x, y):
        y_pred = self.predict(x)
        return (y_pred == y).mean()




    def prediction_growth(self, x, y):
        predictions = {}
        num_nodes = 1
        total_correct = (y == self.val).sum()
        predictions[1] = total_correct

        _, num_nodes = self.correct_growth_helper(x, y, total_correct,
                                                  num_nodes, predictions)
        for n in predictions:
            predictions[n] /= len(y)

        return predictions

    def category_partition_x(self, x):
        idx_sort = np.argsort(x)
        sorted_x = x[idx_sort]
        attrs, idx_start, attrs_cnt = np.unique(sorted_x,
                                                return_index=True,
                                                return_counts=True)
        attrs_idx = np.split(idx_sort, idx_start[1:])
        return attrs, attrs_idx, attrs_cnt

    def category_partition(self, x):
        rand_idx = np.random.randint(len(self.split_val))
        attrs, attrs_idx, _ = self.category_partition_x(x)
        nodes_idx = []
        for i in range(len(attrs)):
            child_idx = self.split_val.get(attrs[i], rand_idx)
            nodes_idx.append(child_idx)
        return nodes_idx, attrs, attrs_idx

    def numerical_partition_x(self, x):
        lr_idx = []
        lr_idx.append(np.nonzero(x <= self.split_val)[0])
        lr_idx.append(np.nonzero(x > self.split_val)[0])
        return lr_idx

    def numerical_partition(self, x):
        lr_idx = self.numerical_partition_x(x)

        nodes_idx, attrs_idx = [], []
        for i, l in enumerate(lr_idx):
            if len(l):
                nodes_idx.append(i)
                attrs_idx.append(l)
        return nodes_idx, attrs_idx

    def correct_growth_helper(self, x, y, total_correct, num_nodes, predictions):
        if not self.is_leaf:
            node_correct = (y == self.val).sum()
            total_correct -= node_correct

            split_col = x[:, self.split_var]
            if self.is_cat_col is not None and self.is_cat_col[self.split_var]:
                children_idx, _, attrs_idx = self.category_partition(split_col)
            else:
                children_idx, attrs_idx = self.numerical_partition(split_col)

            for i in range(len(attrs_idx)):
                total_correct += (y[attrs_idx[i]] == self.children[children_idx[i]].val).sum()

            num_nodes += len(attrs_idx)
            predictions[num_nodes] = total_correct

            for i in range(len(attrs_idx)):
                total_correct, num_nodes = self.children[children_idx[i]].correct_growth_helper(x[attrs_idx[i]],
                                                                  y[attrs_idx[i]],
                                                                  total_correct,
                                                                  num_nodes,
                                                                  predictions)
        return total_correct, num_nodes


    def pruning_result(self, x, y):
        leaf_acc = (y == self.val).mean()
        node_acc = self.accuracy(x, y)

        if leaf_acc >= node_acc:
            self.score = float('inf')
        return self

    def post_pruning(self, x, y):
        if self.is_leaf:
            return self.pruning_result(x, y)
        else:
            tree = copy.copy(self)
            tree.children = [c for c in self.children]

            split_col = x[:, self.split_var]
            if self.is_cat_col is not None and self.is_cat_col[self.split_var]:
                children_idx, _, attrs_idx = self.category_partition(split_col)
            else:
                children_idx, attrs_idx = self.numerical_partition(split_col)

            for i in range(len(attrs_idx)):
                tree.children[children_idx[i]] = self.children[children_idx[i]].post_pruning(x[attrs_idx[i]], y[attrs_idx[i]])

            return tree.pruning_result(x, y)


    @property
    def is_leaf(self): return self.score == float('inf')


    def __repr__(self):
        rep = f"n: {self.n}; val: {self.val}"
        if not self.is_leaf:
            rep += f"; score : {self.score:.4f}; var : {self.split_var}"
        return rep
