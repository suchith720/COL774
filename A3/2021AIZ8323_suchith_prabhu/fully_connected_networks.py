import os
import time
import copy
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from pandas.api.types import is_string_dtype, is_categorical_dtype, is_numeric_dtype

def apply_category(df, trn):
    for n,c in df.items():
        if (n in trn.columns) and (trn[n].dtype.name=='category'):
            df[n] = c.astype('category').cat.as_ordered()
            df[n] = df[n].cat.set_categories(trn[n].cat.categories, ordered=True)

def convert_category(df):
    for n, c in df.items():
        df[n] = c.astype('category').cat.as_ordered()

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
computes f1 score
"""
def F1_score(precision, recall):
    n = 2*precision*recall
    d = precision + recall
    return n/d



"""
computes f1 score from confusion matrix
"""
def compute_F1_from_confusion(conf_df):
    conf_matrix = conf_df.to_numpy()
    f1_scores = []

    for c in range(len(conf_matrix)):
        fp_tp = conf_matrix[:,c].sum()
        if fp_tp:
            precision = conf_matrix[c, c]/fp_tp
        else:
            precision = 1

        recall = conf_matrix[c, c]/conf_matrix[c, :].sum()
        f1 = F1_score(precision, recall)
        f1_scores.append(f1)

    return pd.DataFrame(f1_scores,
                        index=conf_df.columns.to_list(),
                        columns=['F1 score'])



def squared_loss(scores, y):
    num_train = len(y)
    diff = scores.copy()
    diff[range(num_train), y] -= 1

    loss = np.sum(diff**2)
    loss /= 2*num_train

    dscores = diff/num_train
    return loss, dscores


class Linear():

  @staticmethod
  def forward(x, w, b):
    num_train = x.shape[0]
    z = x.reshape(num_train, -1)@w + b
    cache = (x, w, b)
    return z, cache

  @staticmethod
  def backward(dz, cache):
    x, w, b = cache
    num_train = x.shape[0]
    dw = x.reshape(num_train, -1).T@dz
    dx = (dz@w.T).reshape(x.shape)
    db = dz.sum(axis=0)
    return dx, dw, db


class ReLU():

  @staticmethod
  def forward(x):
    a = np.maximum(x, 0)
    cache = x
    return a, cache

  @staticmethod
  def backward(da, cache):
    x = cache
    dx = np.where( x > 0, da, 0)
    return dx


class Sigmoid():

  @staticmethod
  def forward(x):
    a = 1 + np.exp(-x)
    a = np.reciprocal(a)
    cache = a
    return a, cache

  @staticmethod
  def backward(da, cache):
    a = cache
    dx = da*a*(1 - a)
    return dx


class Linear_ReLU():

  @staticmethod
  def forward(x, w, b):
    z, fc_cache = Linear.forward(x, w, b)
    a, relu_cache = ReLU.forward(z)
    cache = (fc_cache, relu_cache)
    return a, cache

  @staticmethod
  def backward(da, cache):
    fc_cache, relu_cache = cache
    dz = ReLU.backward(da, relu_cache)
    dx, dw, db = Linear.backward(dz, fc_cache)
    return dx, dw, db


class Linear_Sigmoid():

  @staticmethod
  def forward(x, w, b):
    z, fc_cache = Linear.forward(x, w, b)
    a, sigmoid_cache = Sigmoid.forward(z)
    cache = (fc_cache, sigmoid_cache)
    return a, cache

  @staticmethod
  def backward(da, cache):
    fc_cache, sigmoid_cache = cache
    dz = Sigmoid.backward(da, sigmoid_cache)
    dx, dw, db = Linear.backward(dz, fc_cache)
    return dx, dw, db



class FullyConnectedNet():

  def __init__(self, hidden_dims, input_dim=85, num_classes=10,
               reg=0.0, weight_scale=1e-2, dtype=np.float,
               activation='sigmoid', sampling='random'):

    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    if activation == 'sigmoid':
      self.activation_fnc = Linear_Sigmoid
    else:
      self.activation_fnc = Linear_ReLU

    if sampling == 'random':
      self.sampling_fnc = self.random_sampling
    else:
      self.sampling_fnc = self.skewed_sampling


    neurons_per_layer = [input_dim] + hidden_dims + [num_classes]
    for i in range(1, len(neurons_per_layer)):
      self.params[f"W{i}"] = weight_scale * np.random.randn( neurons_per_layer[i-1], neurons_per_layer[i]).astype(dtype=self.dtype)
      self.params[f"b{i}"] = np.zeros(neurons_per_layer[i], dtype=self.dtype)


  def loss(self, X, y=None):
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    """
    forward pass through the network
    """
    #hidden layer
    a, la_cache = X, []
    for i in range(1, self.num_layers):
      a, cache = self.activation_fnc.forward(a, self.params[f"W{i}"], self.params[f"b{i}"] )
      la_cache.append(cache)

    #output layer
    scores, cache = Linear_Sigmoid.forward(a, self.params[f"W{self.num_layers}"], self.params[f"b{self.num_layers}"] )
    la_cache.append(cache)

    if mode == 'test':
      return scores


    """
    computing the loss
    """
    loss, grads = 0.0, {}

    loss, dscores = squared_loss(scores, y)
    for i in range(1, self.num_layers+1):
      loss += self.reg * np.sum(self.params[f"W{i}"] * self.params[f"W{i}"])


    """
    backward pass through the network
    """
    #output layer
    da, grads[f"W{self.num_layers}"], grads[f"b{self.num_layers}"] = Linear_Sigmoid.backward(dscores, la_cache[-1])
    grads[f"W{self.num_layers}"] += 2 * self.reg * self.params[f"W{self.num_layers}"]

    #hidden layer
    for i in range(self.num_layers-1, 0, -1):
      da, grads[f"W{i}"], grads[f"b{i}"] = self.activation_fnc.backward(da, la_cache[i-1])
      grads[f"W{i}"] += 2 * self.reg * self.params[f"W{i}"]

    return loss, grads


  def random_sampling(self, X, y, batch_size=100):
    num_train = X.shape[0]
    batch_mask = np.random.permutation(num_train)[:batch_size]
    X_batch = X[batch_mask]
    y_batch = y[batch_mask]
    return X_batch, y_batch


  def skewed_sampling(self, X, y, batch_size=100):
    classes = np.unique(y)
    num_per_class = max(1, int(batch_size/len(classes)) )

    X_class_batch, y_class_batch = [], []
    for c in classes:
        class_pos = np.where(y == c)[0]
        class_mask = np.random.choice(class_pos, num_per_class)

        X_class_batch.append(X[class_mask])
        y_class_batch.append(y[class_mask])

    return np.vstack(X_class_batch), np.hstack(y_class_batch)


  def _step(self, X, y, batch_size=100, lr=1e-2):
    X_batch, y_batch = self.sampling_fnc(X, y, batch_size)

    loss, grads = self.loss(X_batch, y_batch)

    for p, w in self.params.items():
      dw = grads[p]
      next_w = sgd(w, dw, lr)
      self.params[p] = next_w

    return loss


  def predict(self, X):
    scores = self.loss(X)
    y_pred = np.argmax(scores, axis=1)
    return y_pred

  def accuracy(self, X, y):
    scores = self.loss(X)
    y_pred = np.argmax(scores, axis=1)
    acc = (y_pred == y).mean()
    return acc


  def train(self, data, batch_size=100, lr_init=1e-2, num_epochs=10,
            epoch_per_stop=1, verbose=True, use_adaptive_lr=True):

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

    num_train = X_train.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)
    num_iterations = num_epochs * iterations_per_epoch

    loss_history = []
    train_acc_history = []
    val_acc_history = []

    best_val_acc = 0
    best_params = {}

    prev_stop_acc = 0
    avg_stop_acc = 0

    epoch = 1
    lr = lr_init
    for t in range(num_iterations):
      loss = self._step(X_train, y_train, batch_size, lr)
      loss_history.append(loss)

      epoch_end = (t + 1) % iterations_per_epoch == 0
      if epoch_end:
        epoch += 1
        if use_adaptive_lr:
          lr = lr_init/np.sqrt(epoch)


      first_it = t == 0
      last_it = (t == num_iterations - 1)
      if first_it or last_it or epoch_end:
        train_acc = self.accuracy(X_train, y_train)
        val_acc = self.accuracy(X_val, y_val)
        avg_stop_acc += val_acc
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        if verbose:
          print(f"(Epoch {epoch} / {num_epochs}) train acc: {train_acc:.4f}; \
                val_acc: {val_acc:.4f}; loss: {loss_history[-1]:.4f}")

        if val_acc > best_val_acc:
          best_val_acc = val_acc
          for k, v in self.params.items():
            best_params[k] = v.copy()

        stop_check = epoch%epoch_per_stop == 0
        if stop_check:
          avg_stop_acc /= epoch_per_stop
          if avg_stop_acc <= prev_stop_acc:
            break
          prev_stop_acc = avg_stop_acc
          avg_stop_acc = 0


    return loss_history, train_acc_history, val_acc_history, best_params



def sgd(w, dw, lr=1e-2):
    w -= lr * dw
    return w

