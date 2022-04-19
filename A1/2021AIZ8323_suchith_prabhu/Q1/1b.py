from linear import *

def grad_ani(i, line, X, theta_values):
    i = i%len(theta_values)
    line.set_data(X[:, 0],linear(X, theta_values[i]).reshape(-1))
    return line

def get_animation(X_data, Y_data, theta_values):
    n = 100
    X = np.linspace(-2, 4, n)
    X = np.hstack([X.reshape(-1, 1), np.ones((n, 1))])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(X_data[:,0], Y_data)
    ax.set_xlim((-2,4))
    ax.set_ylim((0.99,1.002))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Gradient Descent')
    line, = ax.plot(X[:, 0],linear(X, theta_values[0]).reshape(-1),
                    color="r")

    animator = ani.FuncAnimation(fig, grad_ani, frames=100,
                                 interval=100,
                                 fargs=(line, X, theta_values))
    return animator


from input import *
X_data = X
Y_data = Y


#reading the parameters
thetas = np.load("./data/1a_thetas.npy", allow_pickle=True)
theta = thetas[-1]

n = 100
X = np.linspace(-2, 4, n)
X = np.hstack([X.reshape(-1, 1), np.ones((n, 1))])

plt.figure(figsize=(10, 5))
plt.scatter(X_data[:,0], Y_data)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(X[:, 0],linear(X, theta).reshape(-1), color="r")
plt.show()

animator = get_animation(X_data, Y_data, thetas)
plt.show()

