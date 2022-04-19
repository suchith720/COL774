from input import *
import matplotlib.pyplot as plt

theta = np.load("./data/param.npy")

x = np.linspace(-2, 2, 30)
y = - (x*theta[0]+theta[2])/theta[1]

plt.figure(figsize=(14, 7))
plt.plot(x, y)
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(X_train[:,0:1][Y_train==0], X_train[:,1:2][Y_train==0],
            marker='x')
plt.scatter(X_train[:,0:1][Y_train==1], X_train[:,1:2][Y_train==1])
plt.show()
