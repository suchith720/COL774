from linear import *

from input import *
"""
splitting data into training and validation set and randomly
shuffling the data
"""
valid_pc = 0.8
n_train = int(X.shape[0]*valid_pc)
rnd_idx = np.random.permutation(X.shape[0])
X_train, Y_train = X[rnd_idx[:n_train]], Y[rnd_idx[:n_train]]
X_valid, Y_valid = X[rnd_idx[n_train:]], Y[rnd_idx[n_train:]]

X_data = np.vstack([X_train, X_valid])
Y_data = np.vstack([Y_train, Y_valid])

"""
Training
"""
lrs = [0.001, 0.025, 0.1]
train_losses, valid_losses, theta_values, titles = [], [], [], []
for lr in lrs:
    thetas, train_loss, valid_loss = batch_gradient_descent(X_data,
                                                            Y_data,
                                                            X_valid,
                                                            Y_valid,
                                                            lr=lr,
                                                            eps=1e-2,
                                                            max_iter=10e4)
    theta_values.append(thetas)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    title = f"Learning rate : {lr}"
    titles.append(title)

"""
Plotting contours
"""
time = [1, 100, 100]
frames = [50**3, 200, 100]
for tr, th, t, f, title in zip(train_losses, theta_values, time,frames, titles):
    plot_contour(X_data, Y_data, tr, th, title)
    plot_contour_ani(X_data, Y_data, tr, th, t, f, title)



