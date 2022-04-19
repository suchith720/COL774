from linear import *

def sigmoid(z):
    y = np.exp(-z) + 1
    return 1/y

def logistic(X, theta):
    z = linear(X, theta)
    return sigmoid(z)

def logistic_grad(X, theta, Y):
    Y_hat = logistic(X, theta)
    grad_theta = X.T@(Y - Y_hat)
    return grad_theta

def logistic_loglikelihood(X, theta, Y):
    Y_hat = logistic(X, theta)
    return np.sum(np.log(np.where(Y, Y_hat, 1 - Y_hat)) )

def logistic_hessian(X, theta, Y):
    Y_hat = logistic(X, theta)
    c = -1 * Y_hat * (1 - Y_hat)
    X_tfm = Y_hat * X
    return - X.T @ X_tfm

def newton_descent(X_train, Y_train, eps=1e-6, max_iter=100):
    theta = np.zeros((3, 1))

    theta_values, loglikelihood = [], []
    theta_values.append(theta.copy())
    loglikelihood.append(logistic_loglikelihood(X_train, theta, Y_train))

    num_iter = 0
    while True:
        grad = logistic_grad(X_train, theta, Y_train)
        H = logistic_hessian(X_train, theta, Y_train)
        H_inv = np.linalg.inv(H)
        theta -= H_inv@grad

        theta_values.append(theta.copy())
        loglikelihood.append(logistic_loglikelihood(X_train, theta,
                                                    Y_train))

        num_iter += 1
        if (np.abs(theta_values[-1] - theta_values[-2]) < eps).all() or num_iter > max_iter:
            return loglikelihood, theta_values



from input import *

"""
Visualizing the data
"""
plt.scatter(X_train[:,0:1][Y_train==0], X_train[:,1:2][Y_train==0],
            marker='x')
plt.scatter(X_train[:,0:1][Y_train==1], X_train[:,1:2][Y_train==1])
plt.show()



"""
Training
"""
loglikehood, theta_values = newton_descent(X_train, Y_train, eps=1e-5, max_iter=1000)
plt.xlabel("iterations")
plt.ylabel("log likehood")
plt.plot(loglikehood)
plt.show()

print("Parameters : ")
print(theta_values[-1])

os.makedirs("./data/", exist_ok=True)
np.save("./data/param", theta_values[-1])



