from linear import *
plt.rcParams["figure.figsize"] = (7,7)

from input import *
X_data = X
Y_data = Y


def get_mesh(X_data, Y_data, x_range, y_range):
    x = np.linspace(x_range[0], x_range[1], 30)
    y = np.linspace(y_range[0], y_range[1], 30)

    X, Y = np.meshgrid(x, y)

    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            th = np.array([X[r][c], Y[r][c]]).reshape(-1, 1)
            Z[r][c] = linear_loss(X_data, th, Y_data)

    return X, Y, Z


"""
plotting the loss function
"""
X, Y, Z = get_mesh(X_data, Y_data, (-6, 6), (-6, 6))
ax = plt.axes(projection='3d')
ax.set_title('Loss function');
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.view_init(10, 35)
plt.show()



thetas = np.load("./data/1a_thetas.npy", allow_pickle=True)
train_loss = np.load("./data/1a_loss.npy", allow_pickle=True)
xdata = np.array(thetas)[:, 0, 0]
ydata = np.array(thetas)[:, 1, 0]
zdata = train_loss
"""
Tracking the thetas while training
"""

X, Y, Z = get_mesh(X_data, Y_data, (-1, 1), (-0.1, 2.1))

ax = plt.axes(projection='3d')
ax.set_title('Loss function');
ax.plot_surface(X, Y, Z, alpha=0.8)
ax.plot3D(xdata, ydata, zdata)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('J')
ax.scatter3D(xdata, ydata, zdata, c='r', s=50, depthshade=False)
ax.view_init(40, 10)
plt.show()


"""
Animation
"""
def surf_ani(i, line, points, xdata, ydata, zdata):
    i = i%len(xdata)
    points._offsets3d = (xdata[:i],ydata[:i], zdata[:i])
    line.set_data_3d(xdata[:i], ydata[:i], zdata[:i])
    return line, points


X, Y, Z = get_mesh(X_data, Y_data, (-0.02, 0.02), (-0.1, 1.4))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Loss function');
ax.plot_surface(X, Y, Z, alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('J')
line, = ax.plot3D([], [], [])
points = ax.scatter3D([], [], [], c='r', s=40, depthshade=False)
ax.view_init(40, 10)

animator = ani.FuncAnimation(fig, surf_ani, frames=100, interval=1000,
                             fargs=(line, points, xdata, ydata, zdata))
plt.show()
