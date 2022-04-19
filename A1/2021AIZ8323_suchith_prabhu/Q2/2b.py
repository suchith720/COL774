from linear import *
import pickle

def sgd(X_train, Y_train, bs=100, lr=0.1, k=50,
        eps=1e-6,  max_iter=1000):
    theta = np.zeros((3, 1))

    thetas, train_loss = [], []

    thetas.append(theta.copy())
    train_loss.append(linear_loss(X_train[:bs, :], theta,
                                  Y_train[:bs, :]))

    num_iter = 0
    avg_dtheta = 0

    iter_per_epoch = math.ceil(X_train.shape[0]/bs)

    while True:
        #creating a mini-batch
        bn = num_iter%iter_per_epoch
        b_start = int(bn*bs)
        X_batch, Y_batch = X_train[b_start:b_start+bs, :],Y_train[b_start:b_start+bs, :]

        #descent step
        dtheta = linear_grad(X_batch, theta, Y_batch)
        avg_dtheta += dtheta
        theta -= lr * dtheta

        num_iter += 1

        train_loss.append(linear_loss(X_batch, theta, Y_batch))
        thetas.append(theta.copy())

        #stopping condition
        if num_iter%k == 0:
            avg_dtheta /= k
            if (np.abs(avg_dtheta) < eps).all() or num_iter > max_iter:
                break
            avg_dtheta = 0

    return train_loss, thetas

"""
Loading generated data
"""
X_train = np.load("./data/X_train.npy")
Y_train = np.load("./data/Y_train.npy")


"""
Training
"""
bss = [1, 10**2, 10**4, 10**6]
ks = [10**3, 10**2, 10, 1]
param_file = "./data/params.pickle"

if os.path.exists(param_file):
    file = open(param_file, "rb")
    theta_values, train_losses, timings = pickle.load(file)
    file.close()
else:
    theta_values, train_losses, timings = [], [], []

    for i,(bs,k) in enumerate(zip(bss, ks)):
        print(f"Training with batch size : {bs} :: {k}")
        start_time = time.time()
        train_loss, thetas = sgd(X_train, Y_train, bs=bs, lr=1e-3, k=k,
                                 eps=1e-5, max_iter=10**6)

        theta_values.append(thetas)
        train_losses.append(train_loss)

        end_time = time.time()
        timings.append(end_time-start_time)

    file = open(param_file, "wb")
    pickle.dump((theta_values, train_losses, timings), file)
    file.close()


"""
Printing the parameters
"""
for i,thetas in enumerate(theta_values):
    theta = thetas[-1]
    print(f"Theta {i+1} : {theta[0][0]} {theta[1][0]} {theta[2][0]}")



"""
Sampling loss for plotting loss function
"""
sampled_losses = []
for i, k in enumerate(ks):
    train_loss = train_losses[i]
    sampled_loss = []
    for p in range(0, len(train_loss), k):
        sampled_loss.append(train_loss[p])
    sampled_losses.append(sampled_loss)

fig, axs = plt.subplots(4, 1)
for i, ax in enumerate(axs):
    ax.set_ylim(0, 6)
    ax.set_title(f"batch size : {bss[i]}")
    ax.plot(sampled_losses[i])
plt.show()

