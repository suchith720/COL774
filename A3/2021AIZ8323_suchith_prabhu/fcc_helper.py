import os
import time
import copy
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fully_connected_networks import *
from IPython.display import display
from pandas.api.types import is_string_dtype, is_categorical_dtype, is_numeric_dtype

def split_skewed(X, y, val_pc=0.8):
    classes = np.unique(y)

    X_trains, y_trains, X_vals, y_vals = [], [], [], []
    for c in classes:
        class_pos = np.where(y == c)[0]
        num_train = max(1, int(len(class_pos)*val_pc))

        rand_idx = np.random.permutation(len(class_pos))

        train_pos = class_pos[ rand_idx[:num_train] ]
        X_trains.append(X[train_pos])
        y_trains.append(y[train_pos])

        val_pos = class_pos[ rand_idx[num_train:] ]
        X_vals.append(X[val_pos])
        y_vals.append(y[val_pos])

    data_dict = {}
    data_dict['X_train'] = np.vstack(X_trains)
    data_dict['y_train'] = np.hstack(y_trains)

    data_dict['X_val'] = np.vstack(X_vals)
    data_dict['y_val'] = np.hstack(y_vals)

    return data_dict



def split_random(X, y, val_pc=0.8):
    N = df.shape[0]
    num_train = max(1, int(N*val_pc))

    rand_idx = np.random.permutation(N)

    train_pos = rand_idx[:num_train]
    X_train = X[train_pos]
    y_train = y[train_pos]

    val_pos = rand_idx[num_train:]
    X_val = X[val_pos]
    y_val = y[val_pos]


    data_dict = {}
    data_dict['X_train'] = X_train
    data_dict['y_train'] = y_train

    data_dict['X_val'] = X_val
    data_dict['y_val'] = y_val

    return data_dict


def read_processed_data(save_file):
    if not os.path.exists(save_file):
        print(f"ERROR::File does not exist, {save_file}, run question 2a")
        exit(0)

    with open(save_file, 'rb') as file:
        X_train, X_test, y_train, y_test = pickle.load(file)

    data_dict = split_skewed(X_train, y_train)

    data_dict['X_test'] = X_test
    data_dict['y_test'] = y_test

    return data_dict

def q2c_helper(data_dict, activation='sigmoid', use_adaptive_lr=False,
              epoch_per_stop=10):
    weight_scale = 1e-1
    learning_rate = 20
    batch_size = 100

    hidden_dims = [5, 10, 15, 20, 25]
    print("Training the model : ")
    accuracies, test_pred, val_hist, train_time = [], [], [], []
    for h in hidden_dims:
        model = FullyConnectedNet(hidden_dims=[h], input_dim=85,
                                  num_classes=10,
                                  weight_scale=weight_scale,
                                  activation=activation,
                                  sampling='random')
        start_time = time.time()
        output = model.train(data_dict, batch_size=batch_size,
                             num_epochs=100,
                             epoch_per_stop=epoch_per_stop,
                             lr_init=learning_rate,
                             use_adaptive_lr=use_adaptive_lr,
                             verbose=False)
        end_time = time.time()

        """
        using best model for prediction
        """
        model.params = output[3]

        """
        model accuracy
        """
        a = []
        a.append(model.accuracy(data_dict['X_train'],
                                data_dict['y_train']))
        a.append(model.accuracy(data_dict['X_val'], data_dict['y_val']))
        a.append(model.accuracy(data_dict['X_test'],
                                data_dict['y_test']))
        accuracies.append(a)

        """
        model stats
        """
        train_time.append(end_time-start_time)
        test_pred.append(model.predict(data_dict['X_test']))
        val_hist.append(output[2])


    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    for r in val_hist:
        plt.xlabel('number of iterations')
        plt.ylabel('validation accuracy')
        plt.plot(r)
    plt.title('Validation accuracy over training')
    plt.legend(hidden_dims)
    plt.show()



    print("Accuracy and training time : ")
    display(pd.DataFrame(np.hstack([accuracies, np.array(train_time).reshape(-1, 1)]),
             index=hidden_dims,
             columns=['Train', 'Validation', 'Test', 'Time']).T)
    accuracies = np.array(accuracies)
    for i in range(accuracies.shape[1]):
        plt.plot(accuracies[:, i])
    plt.title('Accuracy')
    plt.legend(['train', 'valid', 'test'])
    plt.show()



    print("Confusion matrix : ")
    y_test = data_dict['y_test']
    for i in range(len(hidden_dims)):
        print(f"Hidden size : {hidden_dims[i]}")
        display(create_confusion_matrix(test_pred[i], y_test,
                                        np.unique(y_test)))

def q2e_helper(data_dict, activation='sigmoid', sampling='random'):
    weight_scale = 1e-1
    learning_rate = 10
    batch_size = 100

    """
    Training model
    """
    acc = []
    model = FullyConnectedNet(hidden_dims=[100, 100], input_dim=85,
                              num_classes=10, weight_scale=weight_scale,
                              activation=activation, sampling=sampling)
    output = model.train(data_dict,
                         batch_size=batch_size, num_epochs=100,
                     epoch_per_stop=10, lr_init=learning_rate,
                     use_adaptive_lr=True, verbose=False)
    loss_history, train_acc_history, val_acc_history, best_params = output
    model.params = best_params

    """
    Plotting loss history
    """
    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    plt.plot(loss_history, 'o')
    plt.xlabel('Iteration')

    plt.subplot(2, 1, 2)
    plt.title('Accuracy')
    plt.plot(train_acc_history, '-o', label='train')
    plt.plot(val_acc_history, '-o', label='val')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    #plt.gcf().set_size_inches(15, 12)
    plt.show()

    """
    Measuring accuracy
    """
    acc = []
    acc.append(model.accuracy(data_dict['X_train'],
                              data_dict['y_train']))
    acc.append(model.accuracy(data_dict['X_val'], data_dict['y_val']))
    acc.append(model.accuracy(data_dict['X_test'], data_dict['y_test']))

    display(pd.DataFrame(np.hstack([[acc]]), index=['sigmoid'] ,
                       columns=['Train', 'Validation', 'Test']))


    """
    Confusion matrix
    """
    test_pred = model.predict(data_dict['X_test'])
    y_test = data_dict['y_test']
    display(create_confusion_matrix(test_pred,
                                    y_test, np.unique(y_test)))

def accuracy(y_pred, y):
    return (y_pred == y).mean()






