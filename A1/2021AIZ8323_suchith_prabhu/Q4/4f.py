from input import *
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 40)
y = np.load("./data/plot_1.npy")
y2 = np.load("./data/plot_2.npy")

plt.figure(figsize=(14, 7))
plt.plot(x, y)
plt.plot(x, y2)
plt.scatter(X_train[:, 0:1][Y_train==classes[0]],
            X_train[:, 1:2][Y_train==classes[0]], marker='x')
plt.scatter(X_train[:, 0:1][Y_train==classes[1]],
            X_train[:, 1:2][Y_train==classes[1]])
plt.show()

