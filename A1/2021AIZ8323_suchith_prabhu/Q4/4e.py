from input import *
import pickle
import matplotlib.pyplot as plt

file = open("./data/param_2.pickle", "rb")
phi, mu_1, mu_2, S_1, S_2 = pickle.load(file)
file.close()

S_1_inv = np.linalg.inv(S_1)
S_2_inv = np.linalg.inv(S_2)

C1 = mu_2.T@S_2_inv@mu_2 - mu_1.T@S_1_inv@mu_1 + 2*np.log((1-phi)/phi) + np.log( np.linalg.det(S_2)/ np.linalg.det(S_1) )

C2 = S_2_inv@mu_2 - S_1_inv@mu_1
C2 = -2*C2

C3 = S_2_inv - S_1_inv

c1 = C3[0, 0]
c2 = C3[0, 1]
c3 = C3[1, 0]
c4 = C3[1, 1]
c5 = C2[0, 0]
c6 = C2[1, 0]
c7 = C1[0, 0]

x = np.linspace(-2, 2, 40)

a = c4
b = (c2+c3)*x + c6
c = c1*x**2 + c5*x + c7

y1 = (-b+np.sqrt(b**2 - 4*a*c))/2*a
y2 = (-b-np.sqrt(b**2 - 4*a*c))/2*a

np.save("./data/plot_2", y1)

plt.figure(figsize=(14, 7))
plt.plot(x, y1)
plt.scatter(X_train[:, 0:1][Y_train==classes[0]],
            X_train[:, 1:2][Y_train==classes[0]], marker='x')
plt.scatter(X_train[:, 0:1][Y_train==classes[1]],
            X_train[:, 1:2][Y_train==classes[1]])
plt.show()


