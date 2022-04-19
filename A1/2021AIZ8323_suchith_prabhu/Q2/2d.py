from linear import *
import pickle

param_file = "./data/params.pickle"
file = open(param_file, "rb")
theta_values, train_losses, timings = pickle.load(file)
file.close()

bss = [1, 10**2, 10**4, 10**6]

fig = plt.figure(figsize=(10, 10))
colors = ['gray', 'red', 'black', 'blue']
for i in range(4):
    plt_id = int(f"22{i+1}")
    ax = fig.add_subplot(plt_id, projection='3d')
    theta = np.array(theta_values[i])
    ax.set_title(f"batch size {bss[i]}")
    ax.plot3D(theta[:,0,0], theta[:,1,0], theta[:,2,0], colors[i])
plt.show()

