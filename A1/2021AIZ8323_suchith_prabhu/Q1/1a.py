from linear import *

"""
batch gradient descent with stopping criteria as number of iterations
"""
def batch_gradient_descent_1(X_train, Y_train, lr=0.1, num_iter=100):

    theta = np.zeros((2, 1))

    theta_values, training_losses = [], []
    theta_values.append(theta.copy())
    training_losses.append(linear_loss(X_train, theta, Y_train))

    for t in range(num_iter):
        theta -= lr * linear_grad(X_train, theta, Y_train)

        training_losses.append(linear_loss(X_train, theta, Y_train))
        theta_values.append(theta.copy())

    return training_losses, theta_values


"""
batch gradient descent with stopping criteria as change in the theta value
"""
def batch_gradient_descent_2(X_train, Y_train, lr=0.1, eps=1e-6):
    theta = np.zeros((2, 1))

    theta_values, training_losses = [], []
    theta_values.append(theta.copy())
    training_losses.append(linear_loss(X_train, theta, Y_train))

    while True:
        theta -= lr * linear_grad(X_train, theta, Y_train)

        training_losses.append(linear_loss(X_train, theta, Y_train))
        theta_values.append(theta.copy())

        if (np.abs( theta_values[-1] - theta_values[-2]) < eps).all():
            break

    return training_losses, theta_values


"""
batch gradient descent with stopping criteria as gradient value
"""
def batch_gradient_descent_3(X_train, Y_train, lr=0.1, eps=1e-6):
    theta = np.zeros((2, 1))

    theta_values, training_losses = [],[]
    theta_values.append(theta.copy())
    training_losses.append(linear_loss(X_train, theta, Y_train))

    while True:
        dtheta = linear_grad(X_train, theta, Y_train)
        theta -= lr * dtheta

        training_losses.append(linear_loss(X_train, theta, Y_train))
        theta_values.append(theta.copy())

        if (np.abs(dtheta) < eps).all():
            break

    return training_losses, theta_values




from input import *


#splitting data into training and validation set and randomly shuffling the data
valid_pc = 0.8
n_train = int(X.shape[0]*valid_pc)
rnd_idx = np.random.permutation(X.shape[0])
X_train, Y_train = X[rnd_idx[:n_train]], Y[rnd_idx[:n_train]]
X_valid, Y_valid = X[rnd_idx[n_train:]], Y[rnd_idx[n_train:]]


#visualizing the data
X_data = np.vstack([X_train, X_valid])
Y_data = np.vstack([Y_train, Y_valid])

plt.figure(figsize=(10, 5))
plt.scatter(X_data[:,0], Y_data)
plt.show()


#training linear regressor
print("Training..")
lr = 0.314
thetas, train_loss, valid_loss = batch_gradient_descent(X_data, Y_data,
                                                        X_valid,
                                                        Y_valid,
                                                        lr=lr,
                                                        eps=1e-10)
plt.plot(train_loss)
plt.show()

print("Parameter : ")
print(thetas[-1])

os.makedirs("./data/", exist_ok=True)
np.save("./data/1a_thetas", thetas, allow_pickle=True)
np.save("./data/1a_loss", train_loss, allow_pickle=True)
