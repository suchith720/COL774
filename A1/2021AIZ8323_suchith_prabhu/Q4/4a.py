from input import *
import pickle
import matplotlib.pyplot as plt

c1_pos = np.where(Y_train == classes[0])[0]
c2_pos = np.where(Y_train == classes[1])[0]

X_c1 = X_train[c1_pos]
X_c2 = X_train[c2_pos]

n = X_train.shape[0]


"""
computing parameters
"""
phi = c1_pos.shape[0]/n

mu_1 = np.mean(X_c1, axis=0).reshape(-1, 1)
mu_2 = np.mean(X_c2, axis=0).reshape(-1, 1)

X_c1_centered = X_c1 - mu_1.T
X_c2_centered = X_c2 - mu_2.T
S = (X_c1_centered.T@X_c1_centered + X_c2_centered.T@X_c2_centered)/n

os.makedirs("./data/", exist_ok=True)
file = open("./data/param.pickle", "wb")
pickle.dump((phi, mu_1, mu_2, S), file)
file.close()


print(f"Phi : {phi}")
print("Mu_0 : ")
print(mu_1)
print("Mu_1 : ")
print(mu_2)
print("Co-variance matrix : ")
print(S)


plt.figure(figsize=(14, 7))
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train==classes[0])
plt.scatter(mu_1[0], mu_1[1], marker='x', s=100)
plt.scatter(mu_2[0], mu_2[1], marker='x', s=100)
plt.show()
