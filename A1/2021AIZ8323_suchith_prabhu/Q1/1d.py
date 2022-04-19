from linear import *
from input import *

X_data = X
Y_data = Y

thetas = np.load("./data/1a_thetas.npy", allow_pickle=True)
train_loss = np.load("./data/1a_loss.npy", allow_pickle=True)

plot_contour(X_data, Y_data, train_loss, thetas)
plot_contour_ani(X_data, Y_data, train_loss, thetas)


