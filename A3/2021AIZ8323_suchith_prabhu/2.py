import argparse
from fcc_helper import *
from fully_connected_networks import *
from sklearn.neural_network import MLPClassifier

plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

"""
Obtaining train, test and validation paths
"""
parser = argparse.ArgumentParser()

parser.add_argument('train_data_path', type=str, help="specifies the \
                    file containing training data")
parser.add_argument('test_data_path', type=str, help="specifies the \
                    file containing the testing data")
parser.add_argument('part_num', type=str, help="part number of the \
                    question [a-d]")
args = parser.parse_args()


train_data_path = args.train_data_path
test_data_path = args.test_data_path
part_num = args.part_num

if not os.path.exists(train_data_path):
    print(f"ERROR::Invalid path, {train_data_path}")
    exit(0)

if not os.path.exists(test_data_path):
    print(f"ERROR::Invalid path, {test_data_path}")
    exit(0)


q2_data_path = './data/q2_data.pickle'

if part_num == 'a':
    """
    Question 2a : One hot encoding
    --------------------------------
    """
    print("-- Question 2a")
    """
    Reading and processing data
    """
    train_raw = pd.read_csv(train_data_path, low_memory=False, header=None)
    test_raw = pd.read_csv(test_data_path, low_memory=False, header=None)

    convert_category(train_raw)
    apply_category(test_raw, train_raw)

    X_train, y_train, _ = proc_df(train_raw, 10, min_n_ord=100)
    X_test, y_test, _ = proc_df(test_raw, 10, min_n_ord=100)

    display(pd.DataFrame([X_train.shape, X_test.shape],
             columns=['rows', 'columns'],
             index=['train', 'test']).T)


    os.makedirs("./data/", exist_ok = True)
    with open(q2_data_path, 'wb') as file:
        pickle.dump((X_train, X_test, y_train, y_test), file)


elif part_num == 'b':
    """
    Question 2b : Implementation
    --------------------------------
    """
    print("-- Question 2b")
    print("Implemented Fully connected neural network, for details see fully_connected_network.py")

elif part_num == 'c':
    """
    Question 2c : Single hidden layer
    --------------------------------
    """
    print("-- Question 2c")

    data_dict = read_processed_data(q2_data_path)
    q2c_helper(data_dict)
elif part_num == 'd':
    """
    Question 2d : Adaptive learning rate
    ------------------------------------
    """
    print("-- Question 2d")

    data_dict = read_processed_data(q2_data_path)
    q2c_helper(data_dict, use_adaptive_lr=True, epoch_per_stop=30)

elif part_num == 'e':
    """
    Question 2e : ReLU activation
    ------------------------------------
    """
    print("-- Question 2e")

    data_dict = read_processed_data(q2_data_path)

    """
    Sigmoid
    """
    print("Sigmoid :")
    q2e_helper(data_dict, activation='sigmoid')

    """
    ReLu
    """
    print("ReLu : ")
    q2e_helper(data_dict, activation='relu')

elif part_num == 'f':
    """
    Question 2f : sklearn
    ------------------------------------
    """
    print("-- Question 2f")

    data_dict = read_processed_data(q2_data_path)

    clf = MLPClassifier(random_state=1, solver='sgd',
                    hidden_layer_sizes=(100, 100),
                        alpha=0, batch_size=100,
                        learning_rate_init=1e-1,
                        learning_rate='adaptive')

    clf.fit(data_dict['X_train'], data_dict['y_train'])

    """
    Accuracy
    """
    accuracies = []
    data_type = ['train', 'val', 'test']
    for t in data_type:
        y_pred = clf.predict(data_dict[f'X_{t}'])
        accuracies.append(accuracy(y_pred, data_dict[f'y_{t}']))
    display(pd.DataFrame([accuracies],
                         columns=['Training', 'Validation', 'Test']))

    """
    Confusion matrix
    """
    y_test = data_dict['y_test']
    y_pred = clf.predict(data_dict[f'X_test'])
    display(create_confusion_matrix(y_pred, y_test, np.unique(y_test)))

elif part_num == 'g':
    """
    Question 2g : skewed sampling
    ------------------------------------
    """
    print("-- Question 2g")

    data_dict = read_processed_data(q2_data_path)
    print("Skewed sampling : ")
    q2e_helper(data_dict, activation='relu', sampling='skewed')

else:
    print("ERROR: Invalid part number")




