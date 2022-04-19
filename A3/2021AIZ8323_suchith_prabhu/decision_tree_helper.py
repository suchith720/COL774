import os
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from pandas.api.types import is_string_dtype, is_categorical_dtype, is_numeric_dtype

def sensitivity(model, param, values, X_train, y_train, X_valid,
                y_valid):
    accuracy = []
    ov = model.get_params()[param]

    for v in values:
        model.set_params(**{param:v})
        model.fit(X_train, y_train)

        accuracy.append((model.score(X_valid, y_valid),model.oob_score_))
    model.set_params(**{param:ov})
    return accuracy

def plot_accuracy(accuracies, params, name, figsize=(14, 7)):
    accuracy = accuracies[name]
    param = params[name]
    acc_valid, acc_oob = list(zip(*accuracy))
    plt.figure(figsize=figsize)
    plt.plot(param, acc_valid)
    plt.plot(param, acc_oob)
    plt.xlabel(name)
    plt.ylabel('accuracy')
    plt.legend(['validation', 'oob'])
    plt.show()

def plot_nodes_acc(acc_growth, title):
    plt.figure(figsize=(14, 7))
    plt.plot(list(acc_growth.keys()), list(acc_growth.values()))
    plt.legend([title])
    plt.xlabel('number of nodes')
    plt.ylabel('accuracy')
    plt.show()
