from input import *
import pickle
import matplotlib.pyplot as plt

file = open("./data/param.pickle", "rb")
phi, mu_1, mu_2, S = pickle.load(file)
file.close()

S_inv = np.linalg.inv(S)
c1 = mu_2.T@S_inv@mu_2 - mu_1.T@S_inv@mu_1 + 2*np.log((1-phi)/phi)
c2 = S_inv@(mu_1 - mu_2)

plt.figure(figsize=(14, 7))
x = np.linspace(-2, 2, 40)
y = - (x*c2[0][0]+c1[0][0])/c2[1][0]

np.save("./data/plot_1", y)

plt.plot(x, y)
plt.scatter(X_train[:, 0:1][Y_train==classes[0]],
            X_train[:, 1:2][Y_train==classes[0]], marker='x')
plt.scatter(X_train[:, 0:1][Y_train==classes[1]],
            X_train[:, 1:2][Y_train==classes[1]])
plt.show()

