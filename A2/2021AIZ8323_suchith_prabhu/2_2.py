from multiclass_svm import *
"""
Obtaining training and testing paths
"""
parser = argparse.ArgumentParser()

parser.add_argument('train_data_path', type=str, help="specifies the \
                    file containing training data")
parser.add_argument('test_data_path', type=str, help="specifies the \
                    file containing the testing data")
parser.add_argument('part_num', type=str, help="part number of the \
                    question")
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



"""
Reading train and test data
"""
X_train, Y_train, X_test, Y_test = load_mnist(train_data_path,
                                              test_data_path)

X_test_3, Y_test_3 = two_class_data(X_test, Y_test, 3, 4)

#For storing output features
os.makedirs("./data/", exist_ok=True)

#parameter C, threshold
C = 1.0
gamma = 0.05
threshold = 1e-6


if part_num == 'a':

    """
    Multiclass
    """
    print("-- Question 2b. i", end='\n\n')

    svm = OvO_GSVM()
    start = time.time()
    svm.train(X_train, Y_train, X_test, Y_test, C, gamma)
    end = time.time()
    print(f"Train time : {end-start} secs")

    Y_pred_ovo = svm.predict(X_test, threshold)
    acc = accuracy(Y_pred_ovo, Y_test.reshape(-1, 1))
    print(f"Test Accuracy : {acc}")

    with open('./data/y_pred_ovo.pickle', 'wb') as f:
        pickle.dump(Y_pred_ovo, f)


elif part_num == 'b':

    """
    LIBSVM Multiclass
    """
    print("-- Question 2b. ii", end='\n\n')

    prob  = svm_problem(Y_train.reshape(-1), X_train)
    param = svm_parameter('-t 2 -c 1 -g 0.05 -q')

    start = time.time()
    m = svm_train(prob, param)
    end = time.time()
    print(f"Train time : {end-start} secs")

    Y_pred_libovo, p_acc, p_vals = svm_predict(Y_test.reshape(-1),
                                          X_test, m, options='-q')
    print(f"Test Accuracy : {p_acc[0]}")

    with open('./data/y_pred_libovo.pickle', 'wb') as f:
        pickle.dump(Y_pred_libovo, f)


elif part_num == 'c':
    classes = np.unique(Y_train)
    """
    Confusion matrix and visualization
    """
    print("-- Question 2b. iii", end='\n\n')

    confusion_matrix_and_visualize(X_test_3, Y_test_3,
                                   './data/y_pred_l.pickle',
                                   "- Binary linear CVOPTX")

    confusion_matrix_and_visualize(X_test_3, Y_test_3,
                                   './data/y_pred_g.pickle',
                                   "- Binary gaussian CVOPTX")

    confusion_matrix_and_visualize(X_test_3, Y_test_3,
                                   './data/y_pred_libl.pickle',
                                   "- Binary linear LIBSVM")

    confusion_matrix_and_visualize(X_test_3, Y_test_3,
                                   './data/y_pred_libg.pickle',
                                   "- Binary gaussian LIBSVM")

    """
    if not os.path.exists('./data/y_pred_ovo.pickle') or not os.path.exists('./data/y_pred_libovo.pickle'):
        print("ERROR: Run 2bi and 2bii ")
        exit(0)

    #Multiclass CVOPTX

    print("- Multiclass CVOPTX", end='\n\n')
    with open('./data/y_pred_ovo.pickle', 'rb') as f:
        Y_pred_ovo = pickle.load(f)

    #Confusion matrix
    conf_df = create_confusion_matrix(Y_pred_ovo, Y_test, classes)
    print("Confusion Matrix")
    display(conf_df)

    #Most misclassified
    conf_matrix = conf_df.to_numpy()
    print(f"Most misclassified : {most_misclassified(conf_matrix)}")

    visualize_misclassified(X_test, Y_test, Y_pred_ovo, num=10, cols=5)
    """
    confusion_matrix_and_visualize(X_test, Y_test,
                                   './data/y_pred_ovo.pickle',
                                   "- Multiclass CVOPTX")


    """
    #Multiclass LIBSVM
    print("- Multiclass LIBSVM", end='\n\n')
    with open('./data/y_pred_libovo.pickle', 'rb') as f:
        Y_pred_libovo = pickle.load(f)

    #Confusion matrix
    Y_pred_libovo = np.array(Y_pred_libovo, dtype=int).reshape(-1, 1)
    conf_df = create_confusion_matrix(Y_pred_libovo, Y_test, classes)
    print("Confusion Matrix")
    display(conf_df)

    #Most misclassified
    conf_matrix = conf_df.to_numpy()
    print(f"Most misclassified : {most_misclassified(conf_matrix)}")

    visualize_misclassified(X_test, Y_test, Y_pred_libovo,
                            num=10, cols=5)
    """
    confusion_matrix_and_visualize(X_test, Y_test,
                                   './data/y_pred_libovo.pickle',
                                   "- Multiclass LIBSVM")

elif part_num == 'd':
    """
    K fold Validation
    """
    print("-- Question 2b. iv", end='\n\n')
    C = [1e-5, 1e-3, 1, 5, 10]
    K = 5

    print("K fold cross validation")
    if not os.path.exists('./data/kfold_ovo_validation.pickle'):
        accuracies = Kfold_validation_C(X_train, Y_train, C=C, K=5)

        with open('./data/kfold_ovo_validation.pickle', 'wb') as f:
            pickle.dump(accuracies, f)
    else:
        with open('./data/kfold_ovo_validation.pickle', 'rb') as f:
            accuracies = pickle.load(f)


    """
    K fold test data accuracy
    """
    print("Test set accuracy")
    if not os.path.exists('./data/kfold_ovo_test.pickle'):
        test_accuracies = test_accuracy_C(X_train, Y_train,
                                          X_test, Y_test, C)

        with open('./data/kfold_ovo_test.pickle', 'wb') as f:
            pickle.dump(test_accuracies, f)
    else:
        with open('./data/kfold_ovo_test.pickle', 'rb') as f:
            test_accuracies = pickle.load(f)


    #plotting the accuracies
    plt.figure(figsize=(10, 7))
    x, y = list(zip(*accuracies.items()))
    m_acc = np.mean(y, axis=1)
    s_acc = np.std(y, axis=1)
    plt.errorbar(np.log10(x), m_acc, yerr=s_acc)

    x_s = np.log10([x]*K).T
    plt.scatter(x_s, y, color='r')

    x_t, y_t = list(zip(*test_accuracies.items()))
    plt.plot(np.log10(x_t), y_t)
    plt.xlabel("log(C)")
    plt.ylabel("accuracy")
    plt.legend(["Test", "fold accuracy", "K fold accuracy"])
    plt.show()

else:
    print("Invalid part number")


