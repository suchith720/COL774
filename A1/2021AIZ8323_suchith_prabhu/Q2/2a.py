from linear import *
plt.rcParams["figure.figsize"] = (7, 7)

np.random.seed(10)

"""
Creating dataset
"""
n = 10**6
x1 = np.random.randn(n, 1)
x2 = np.random.randn(n, 1)
e = np.random.randn(n, 1)

x1 = 2*x1 + 3
x2 = 2*x2 - 1
e = np.sqrt(2)*e

X_train = np.hstack([x2, x1, np.ones((n, 1))])
theta = np.array([2, 1, 3]).reshape(-1, 1)
Y_train = linear(X_train, theta) + e

os.makedirs("./data/", exist_ok=True)
np.save("./data/X_train", X_train)
np.save("./data/Y_train", Y_train)

"""
Visualizing the dataset
"""
X = X_train[:, 0]
Y = X_train[:, 1]
Z = Y_train[:, 0]
ax = plt.axes(projection='3d')
ax.set_title('surface');
ax.scatter3D(X, Y, Z, c='r', depthshade=False, s=1)
ax.view_init(10, 30)
plt.show()
