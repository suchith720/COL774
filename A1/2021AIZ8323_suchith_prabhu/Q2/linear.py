import os
import math
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.animation as ani

rc('animation', html='html5')
plt.rcParams["figure.figsize"] = (14,7)


def linear(X, theta):
    return X@theta

def linear_loss(X, theta, Y):
    n = X.shape[0]

    Y_hat = linear(X, theta)
    l = (Y_hat - Y).T @ (Y_hat - Y)
    l /= (2*n)

    return l[0][0]

def linear_grad(X, theta, Y):
    n = X.shape[0]
    Y_hat = linear(X, theta)
    grad_theta = X.T@(Y_hat - Y)
    grad_theta /= n
    return grad_theta

"""
batch gradient descent with stopping criteria as gradient value and number of iteration
"""
def batch_gradient_descent(X_train, Y_train, X_valid, Y_valid, lr=0.1,
                           eps=1e-2, max_iter=100):

    #initializing theta to zero
    theta = np.zeros((2, 1))

    thetas, train_loss, valid_loss = [], [], []
    thetas.append(theta.copy())
    train_loss.append(linear_loss(X_train, theta, Y_train))
    valid_loss.append(linear_loss(X_valid, theta, Y_valid))

    num_iter = 0
    while True:
        #descent step
        dtheta = linear_grad(X_train, theta, Y_train)
        theta -= lr * dtheta

        train_loss.append(linear_loss(X_train, theta, Y_train))
        valid_loss.append(linear_loss(X_valid, theta, Y_valid))
        thetas.append(theta.copy())
        num_iter += 1

        #stopping condition
        if (np.abs(dtheta) < eps).all() or max_iter < num_iter:
            break

    return thetas, train_loss, valid_loss


def plot_contour(X_data, Y_data, train_loss, thetas, title="Plot"):
   x = np.linspace(-1, 1, 30)
   y = np.linspace(-0.2, 2, 30)

   X, Y = np.meshgrid(x, y)

   Z = np.zeros(X.shape)
   for r in range(X.shape[0]):
       for c in range(X.shape[1]):
           th = np.array([X[r][c], Y[r][c]]).reshape(-1, 1)
           Z[r][c] = linear_loss(X_data, th, Y_data)

   xdata = np.array(thetas)[:, 0, 0]
   ydata = np.array(thetas)[:, 1, 0]
   zdata = train_loss

   plt.figure()
   plt.title(title)
   plt.contour(X, Y, Z, 40, cmap='RdGy')
   plt.scatter(xdata, ydata, s=10)
   plt.plot(xdata, ydata)
   plt.show()


def contour_ani(i, points, line, xdata, ydata):
    i = i%len(xdata)
    if i == 0:
        i = 1
    points.set_offsets(list(zip(xdata[:i],ydata[:i])))
    line.set_data(xdata[:i], ydata[:i])
    return points, line

def plot_contour_ani(X_data, Y_data, train_loss, thetas,
                     interval=1000, frames=100, title="Plot"):
   x = np.linspace(-1, 1, 30)
   y = np.linspace(-0.2, 2, 30)

   X, Y = np.meshgrid(x, y)

   Z = np.zeros(X.shape)
   for r in range(X.shape[0]):
       for c in range(X.shape[1]):
           th = np.array([X[r][c], Y[r][c]]).reshape(-1, 1)
           Z[r][c] = linear_loss(X_data, th, Y_data)

   xdata = np.array(thetas)[:, 0, 0]
   ydata = np.array(thetas)[:, 1, 0]
   zdata = train_loss

   fig, ax = plt.subplots()
   ax.set_title(title)
   ax.contour(X, Y, Z, 40, cmap='RdGy')
   points = ax.scatter([], [], s=10)
   line, = ax.plot([], [])
   animator = ani.FuncAnimation(fig, contour_ani, frames=frames,
                                interval=interval,
                                fargs=(points, line, xdata, ydata))
   plt.show()

