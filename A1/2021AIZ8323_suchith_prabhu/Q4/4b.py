from input import *
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.scatter(X_train[:, 0:1][Y_train==classes[0]], X_train[:, 1:2][Y_train==classes[0]], marker='x')
plt.scatter(X_train[:, 0:1][Y_train==classes[1]], X_train[:, 1:2][Y_train==classes[1]])
plt.show()

