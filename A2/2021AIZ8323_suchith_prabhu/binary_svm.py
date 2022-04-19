import os
import re
import nltk
import time
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import mode
from scipy.spatial.distance import pdist, squareform, cdist
from libsvm.svmutil import *
from cvxopt import matrix
from cvxopt import solvers

import argparse
from IPython.display import display

plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

solvers.options['show_progress'] = False

"""
loads the mnist data
"""
def load_mnist(train_data_path, test_data_path):
    train_data = pd.read_csv(train_data_path,
                             header=None).to_numpy()
    X_train = train_data[:, :-1]/255
    Y_train = train_data[:, -1].reshape(-1,1)

    test_data = pd.read_csv(test_data_path,
                            header=None).to_numpy()
    X_test = test_data[:, :-1]/255
    Y_test = test_data[:, -1].reshape(-1,1)

    return X_train, Y_train, X_test, Y_test


"""
return data of the two numbers from the mnist dataset
"""
def two_class_data(X, Y, i, j):
    pos_ij = np.where(np.logical_or(Y == i, Y == j))[0]

    X_ij = X[pos_ij]
    Y_ij = np.where( Y[pos_ij] == i, 1, -1)
    return X_ij, Y_ij


"""
Confusion matrix
"""
def create_confusion_matrix(Y_pred, Y, classes):
    confusion_matrix = []

    for class_1 in classes:
        Y_class = Y_pred[Y==class_1]
        class_count = []
        for class_2 in classes:
            class_count.append(np.sum(Y_class == class_2))
        confusion_matrix.append(class_count)

    return pd.DataFrame(confusion_matrix,
                        index=pd.Index(classes, name='Actual :'),
                        columns=pd.Index(classes, name='Predicted :') )

"""
To find out most miss classfied data
"""
def most_misclassified(conf_matrix):
    mask = np.ones(conf_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, 0)

    max_count = np.max(conf_matrix[mask])
    return np.where(conf_matrix == max_count)



def visualize_misclassified(X, y, y_pred,
                            num=10, cols=5, figsize=(10, 5)):

    misclassified, _ = np.where(y != y_pred)
    pos = np.random.permutation(len(misclassified))[:num]
    pos = misclassified[pos]

    rows = int(np.ceil(num/cols))

    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for r in range(rows):
        for c in range(cols):
            cnt = r* rows + c
            if  cnt < len(pos):
                ax[r, c].imshow(X[pos[cnt]].reshape(28, 28),
                                cmap="gray")
                ax[r, c].set_title(f"{y[pos[cnt]][0]} -> {y_pred[pos[cnt]][0]}")
            ax[r, c].axis('off')
    plt.show()


"""
Expressing the classification as QP problem
"""

def QP_coefficients(X_train, Y_train, kernel_func, C=1.0):
    K = kernel_func(X_train)
    YTY = Y_train@Y_train.T

    P = matrix(YTY*K, tc='d')
    N = X_train.shape[0]
    q = matrix(np.full(N, -1), tc='d')

    G1 = np.diag(np.full(N, -1))
    h1 = np.zeros(N)
    G2 = np.eye(N)
    h2 = np.full(N, C)
    G = matrix(np.vstack([G1, G2]), tc='d')
    h = matrix(np.hstack([h1, h2]), tc='d')

    A = matrix(Y_train.T, tc='d')
    b = matrix(0, tc='d')
    return P, q, G, h, A, b


"""
Model Accuracy
"""
def accuracy(Y_pred, Y):
    return np.mean(Y_pred == Y)


"""
Linear SVM
"""
class LinearSVM:

    def __init__(self):
        self.alpha = None
        self.X = None
        self.y = None
        self.c = None

    def linear_kernel(self, X):
        return X@X.T

    def train(self, X_train, Y_train, C=1.0):
        self.X = X_train
        self.y = Y_train
        self.c = C

        P, q, G, h, A, b = QP_coefficients(X_train, Y_train,
                                           self.linear_kernel, C)
        sol = solvers.qp(P, q, G, h, A, b)
        self.alpha = np.array(sol['x'])


    def get_coeff_sv(self, threshold):
        sv_pos, _ = np.where(self.alpha > threshold)
        alpha_sv = self.alpha[sv_pos]
        X_sv = self.X[sv_pos]
        return alpha_sv, X_sv


    def compute_Wb(self, threshold):
        sv_pos, _ = np.where(self.alpha > threshold)

        alpha_sv = self.alpha[sv_pos]
        X_sv = self.X[sv_pos]
        Y_sv = self.y[sv_pos]
        W = np.sum(Y_sv*alpha_sv*X_sv, axis=0, keepdims=True).T

        msv_pos, _ = np.where(alpha_sv < self.c - threshold)
        X_msv = X_sv[msv_pos]
        Y_msv = Y_sv[msv_pos]
        alpha_msv = alpha_sv[msv_pos]
        b = np.mean(Y_msv - X_msv@W)
        return W, b


    def predict(self, X_test, threshold=1e-6):
        W, b = self.compute_Wb(threshold)
        Y_score = X_test@W + b
        Y_pred = np.where(Y_score > 0, 1, -1)
        return Y_pred




"""
Gaussian SVM
"""

class GaussianSVM:

    def __init__(self):
        self.alpha = None
        self.X = None
        self.y = None
        self.c = None

    def gaussian_kernel_fast(self, X, Y, gamma=0.05):
        sq_dists = cdist(X, Y, 'sqeuclidean')
        K = np.exp(-gamma*sq_dists)
        return K

    def gaussian_kernel(self, X, gamma=0.05):
        sq_dists = squareform(pdist(X, 'sqeuclidean'))
        K = np.exp(-gamma*sq_dists)
        return K

    def train(self, X_train, Y_train, C=1.0, gamma=0.05):
        self.X = X_train
        self.y = Y_train
        self.c = C

        kernel = lambda X: self.gaussian_kernel(X, gamma)
        P, q, G, h, A, b = QP_coefficients(X_train, Y_train,
                                           kernel, C)

        sol = solvers.qp(P, q, G, h, A, b)
        self.alpha = np.array(sol['x'])

    def get_coeff_sv(self, threshold):
        sv_pos, _ = np.where(self.alpha > threshold)
        alpha_sv = self.alpha[sv_pos]
        X_sv = self.X[sv_pos]
        return alpha_sv, X_sv

    def predict(self, X_test, threshold=1e-6):
        sv_pos, _ = np.where(self.alpha > threshold)
        alpha_sv = self.alpha[sv_pos]
        X_sv = self.X[sv_pos]
        Y_sv = self.y[sv_pos]

        msv_pos, _ = np.where(alpha_sv < self.c - threshold)
        X_msv = X_sv[msv_pos]
        Y_msv = Y_sv[msv_pos]
        alpha_msv = alpha_sv[msv_pos]

        """
        compute b
        """
        p1 = np.where(Y_msv == 1)[0][0]
        p2 = np.where(Y_msv == -1)[0][0]
        K = self.gaussian_kernel_fast(X_sv, X_msv[[p1, p2]])
        b = - np.sum(alpha_sv*Y_sv*K)/2

        """
        computing prediction
        """
        K = self.gaussian_kernel_fast(X_sv, X_test)
        Y_score = np.sum(alpha_sv*Y_sv*K, axis=0) + b
        Y_score = Y_score.reshape(-1, 1)
        Y_pred = np.where(Y_score > 0, 1, -1)
        return Y_pred, Y_score



