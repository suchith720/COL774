import pickle
import argparse
import numpy as np
import pandas as pd
from IPython.display import display

from linear import *

def l2_error(Y_pred, Y):
    d = (Y_pred - Y)
    l2 = d.T@d
    l2 = l2/Y.shape[0]
    return l2[0][0]

"""
Input for testing
"""
parser = argparse.ArgumentParser()

parser.add_argument('data_path', type=str, help="specifies the \
                    file containing test data")
args = parser.parse_args()
data_path = args.data_path


param_file = "./data/params.pickle"
file = open(param_file, "rb")
theta_values, train_losses, timings = pickle.load(file)
file.close()

"""
comparison
"""
theta = np.array([2, 1, 3]).reshape(-1, 1)
ths, errs, iters = [], [], []
for i,thetas in enumerate(theta_values):
    ths.append(thetas[-1])
    errs.append(np.linalg.norm(ths[-1] - theta))
    iters.append(len(train_losses[i]))
ths = np.array(ths)

df = pd.DataFrame(np.array([ths[:,0,:][:,0], ths[:,1,:][:,0], ths[:,2,:][:,0], errs, iters, timings]).T,
            columns=["theta_2", "theta_1", "theta_0", "l2 error", "iterations", "timing"])

display(df)



"""
Testing
"""

df = pd.read_csv(data_path)
X_test = df[['X_2', 'X_1']].to_numpy()
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
Y_test = df['Y'].to_numpy().reshape(-1, 1)

#computing error
error = []
for th in np.vstack([ths, [theta]]):
    Y_pred = linear(X_test, th)
    l2 = l2_error(Y_pred, Y_test)
    error.append(l2)

df = pd.DataFrame(np.array(error).reshape(1, -1), index=["l2_error"],
             columns=[0, 1, 2, 3, "original"])
display(df)
