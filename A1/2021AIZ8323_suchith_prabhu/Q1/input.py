import argparse
import os
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('X_data_path', type=str, help="specifies the \
                    directory containing X data")
parser.add_argument('Y_data_path', type=str, help="specifies the \
                    directory containing the Y data")
args = parser.parse_args()


X_data_path = args.X_data_path
Y_data_path = args.Y_data_path

if not os.path.exists(X_data_path):
    print(f"ERROR::Invalid path, {X_data_path}")
    exit(0)

if not os.path.exists(Y_data_path):
    print(f"ERROR::Invalid path, {Y_data_path}")
    exit(0)


# read the data and labels
print("Reading data..")
X = pd.read_csv(X_data_path, header=None).to_numpy()
Y = pd.read_csv(Y_data_path, header=None).to_numpy()

# normalizing the data
X = (X - X.mean(axis=0))/X.std(axis=0)

# adding intercept to X_train
X = np.hstack([X, np.ones( (X.shape[0], 1) )])
