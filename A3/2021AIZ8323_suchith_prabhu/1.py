import argparse
from decision_tree import *
from decision_tree_helper import *
from sklearn.ensemble import RandomForestClassifier

"""
Obtaining train, test and validation paths
"""
parser = argparse.ArgumentParser()

parser.add_argument('train_data_path', type=str, help="specifies the \
                    file containing training data")
parser.add_argument('test_data_path', type=str, help="specifies the \
                    file containing the testing data")
parser.add_argument('valid_data_path', type=str, help="specifies the \
                    file containing the validation data")
parser.add_argument('part_num', type=str, help="part number of the \
                    question [a-d]")
args = parser.parse_args()


train_data_path = args.train_data_path
test_data_path = args.test_data_path
valid_data_path = args.valid_data_path
part_num = args.part_num

if not os.path.exists(train_data_path):
    print(f"ERROR::Invalid path, {train_data_path}")
    exit(0)

if not os.path.exists(test_data_path):
    print(f"ERROR::Invalid path, {test_data_path}")
    exit(0)

if not os.path.exists(valid_data_path):
    print(f"ERROR::Invalid path, {valid_data_path}")
    exit(0)


"""
Reading and processing data
"""

train_raw = pd.read_csv(train_data_path, low_memory=False, sep=';')
valid_raw = pd.read_csv(valid_data_path, low_memory=False, sep=';')
test_raw = pd.read_csv(test_data_path, low_memory=False, sep=';')

train_category(train_raw)
apply_category(test_raw, train_raw)
apply_category(valid_raw, train_raw)

if part_num == 'a':
    """
    Question 1a : Decision tree
    --------------------------------
    """
    print("-- Question 1a")

    print("Decision tree using attribute values")
    X_train, y_train, is_cat_col = proc_df(train_raw, y_label='y')
    X_valid, y_valid, _ = proc_df(valid_raw, y_label='y')
    X_test, y_test, _ = proc_df(test_raw, y_label='y')

    tree = DecisionTree(X_train, y_train, is_cat_col)

    display(pd.DataFrame([[tree.accuracy(X_train, y_train),
               tree.accuracy(X_valid, y_valid),
               tree.accuracy(X_test, y_test)]],
             columns=['train', 'validation', 'test'], index=['accuracy']))

    train_acc_growth = tree.prediction_growth(X_train, y_train)
    valid_acc_growth = tree.prediction_growth(X_valid, y_valid)
    test_acc_growth = tree.prediction_growth(X_test, y_test)
    plot_nodes_acc(train_acc_growth, 'train')
    plot_nodes_acc(valid_acc_growth, 'validation')
    plot_nodes_acc(test_acc_growth, 'test')


    print("Decision tree using one hot encoding")
    X_train, y_train, is_cat_col = proc_df(train_raw, y_label='y', min_n_ord=100)
    X_valid, y_valid, _ = proc_df(valid_raw, y_label='y', min_n_ord=100)
    X_test, y_test, _ = proc_df(test_raw, y_label='y', min_n_ord=100)

    tree = DecisionTree(X_train, y_train, is_cat_col)

    display(pd.DataFrame([[tree.accuracy(X_train, y_train),
               tree.accuracy(X_valid, y_valid),
               tree.accuracy(X_test, y_test)]],
             columns=['train', 'validation', 'test'], index=['accuracy']))

    train_acc_growth = tree.prediction_growth(X_train, y_train)
    valid_acc_growth = tree.prediction_growth(X_valid, y_valid)
    test_acc_growth = tree.prediction_growth(X_test, y_test)
    plot_nodes_acc(train_acc_growth, 'train')
    plot_nodes_acc(valid_acc_growth, 'validation')
    plot_nodes_acc(test_acc_growth, 'test')

elif part_num == 'b':
    """
    Question 1b : Pruning decision tree
    --------------------------------
    """
    print("-- Question 1b")
    X_train, y_train, is_cat_col = proc_df(train_raw, y_label='y')
    X_valid, y_valid, _ = proc_df(valid_raw, y_label='y')
    X_test, y_test, _ = proc_df(test_raw, y_label='y')

    tree = DecisionTree(X_train, y_train, is_cat_col)

    pruned_tree = tree.post_pruning(X_valid, y_valid)

    display(pd.DataFrame([[tree.accuracy(X_train, y_train),
               tree.accuracy(X_valid, y_valid),
               tree.accuracy(X_test, y_test)],
              [pruned_tree.accuracy(X_train, y_train),
               pruned_tree.accuracy(X_valid, y_valid),
               pruned_tree.accuracy(X_test, y_test)]],
             columns=['train', 'validation', 'test'],
             index=['Tree', 'Pruned tree']))

    train_acc_growth = pruned_tree.prediction_growth(X_train, y_train)
    valid_acc_growth = pruned_tree.prediction_growth(X_valid, y_valid)
    test_acc_growth = pruned_tree.prediction_growth(X_test, y_test)
    plot_nodes_acc(train_acc_growth, 'train')
    plot_nodes_acc(valid_acc_growth, 'validation')
    plot_nodes_acc(test_acc_growth, 'test')

elif part_num == 'c':
    """
    Question 1c : Random forest sklearn
    --------------------------------
    """
    print("-- Question 1c")

    X_train, y_train, is_cat_col = proc_df(train_raw, y_label='y', min_n_ord=6)
    X_valid, y_valid, _ = proc_df(valid_raw, y_label='y', min_n_ord=6)
    X_test, y_test, _ = proc_df(test_raw, y_label='y', min_n_ord=6)

    model = RandomForestClassifier(n_jobs=-1, n_estimators=40,
                                   max_features=0.5,
                                   min_samples_leaf=5,
                                   oob_score=True).fit(X_train, y_train)

    print("Accuracy on default parameters:")
    display(pd.DataFrame([[model.score(X_train, y_train),
                           model.score(X_valid, y_valid),
                           model.score(X_test, y_test),
                           model.oob_score_]],
                         columns=['Train', 'Validation', 'Test', 'OOB']))


    """
    Grid Search through parameters.
    """
    print("Grid Search through parameters")
    accuracies = {}
    n_estimators = np.arange(50, 451, 100)
    max_features = np.arange(0.1, 1, 0.2)
    min_samples_split = np.arange(2, 10, 2)

    best_model, best_score, best_param = None, None, None
    for ne in n_estimators:
        for mf in max_features:
            for mss in min_samples_split:
                model = RandomForestClassifier(n_jobs=-1, n_estimators=ne,
                                               max_features=mf,
                                               min_samples_leaf=5,
                                               min_samples_split=mss,
                                               oob_score=True,
                                               random_state=125).fit(X_train,
                                                                     y_train)
                accuracies[(ne, mf, mss)] = (model.score(X_valid, y_valid),model.oob_score_)

                if best_score is None or model.oob_score_ > best_score:
                    best_score = model.oob_score_
                    best_param = (ne, mf, mss)
                    best_model = model

    print("Best parameters : ")
    display(pd.DataFrame([best_param], columns=['n_estimators', 'max_features', 'min_samples_split']))
    print("Accuracy on best parameters:")
    display(pd.DataFrame([[best_model.score(X_train, y_train),
                           best_model.score(X_valid, y_valid),
                           best_model.score(X_test, y_test),
                           best_model.oob_score_]],
                         columns=['Train', 'Validation', 'Test', 'OOB']))


    """
    Visualizing the grid seach
    """
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='3d')

    ax.set_xlabel('n_estimators')
    ax.set_ylabel('max_features')
    ax.set_zlabel('min_samples_split')

    c1, c2 = list(zip(*list(accuracies.values())))
    x, y, z = list(zip(*list(accuracies.keys())))

    ax.scatter(best_param[0], best_param[1], best_param[2], s=100, c='r')
    ax.set_title('OOB accuracy')
    p = ax.scatter(x, y, z, c=c2)
    fig.colorbar(p, ax=ax)
    ax.view_init(20, 35)
    plt.show()

elif part_num == 'd':
    """
    Question 1d : Random forest parameter sensitivity
    --------------------------------
    """
    print("-- Question 1d")
    X_train, y_train, is_cat_col = proc_df(train_raw, y_label='y', min_n_ord=6)
    X_valid, y_valid, _ = proc_df(valid_raw, y_label='y', min_n_ord=6)
    X_test, y_test, _ = proc_df(test_raw, y_label='y', min_n_ord=6)

    model = RandomForestClassifier(n_jobs=-1, n_estimators=400,
                                           max_features=0.7,
                                           min_samples_leaf=5,
                                           min_samples_split=2,
                                           oob_score=True,
                                           random_state=125).fit(X_train,
                                                                 y_train)

    accuracies = {}
    params = {}
    params['n_estimators'] = np.arange(50, 450, 100)
    params['max_features'] = np.arange(0.1, 1, 0.2)
    params['min_samples_split'] = np.arange(2, 10, 2)

    for p, v in params.items():
        accuracies[p] = sensitivity(model, p, v, X_train, y_train,
                                    X_valid, y_valid)

    plot_accuracy(accuracies, params, 'n_estimators')
    plot_accuracy(accuracies, params, 'max_features')
    plot_accuracy(accuracies, params, 'min_samples_split')

else:
    print("ERROR: Invalid part number")



