from binary_svm import *
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

X_train_3, Y_train_3 = two_class_data(X_train, Y_train, 3, 4)
X_test_3, Y_test_3 = two_class_data(X_test, Y_test, 3, 4)

#parameter C, threshold
C = 1.0
gamma = 0.05
threshold = 1e-6


if part_num == 'a':

    """
    Linear SVM
    """
    print("-- Question 2a. i", end='\n\n')


    svm = LinearSVM()
    start = time.time()
    svm.train(X_train_3, Y_train_3, C)
    end = time.time()
    print(f"Training time : {end-start} secs")

    Y_pred_train = svm.predict(X_train_3, threshold=threshold)
    train_acc = accuracy(Y_pred_train, Y_train_3)
    print(f"Train Accuracy : {train_acc}")

    Y_pred_test = svm.predict(X_test_3, threshold=threshold)
    test_acc = accuracy(Y_pred_test, Y_test_3)
    print(f"Test Accuracy : {test_acc}")

    alpha_sv, X_sv = svm.get_coeff_sv(threshold=1e-6)
    print(f"Number of SV : {len(alpha_sv)}")

    with open('./data/y_pred_l.pickle', 'wb') as f:
        pickle.dump(Y_pred_test, f)

elif part_num == 'b':

    """
    Gaussian SVM
    """
    print("-- Question 2b. ii", end='\n\n')

    svm = GaussianSVM()
    start = time.time()
    svm.train(X_train_3, Y_train_3, C, gamma)
    end = time.time()
    print(f"Training time : {end-start} secs")

    Y_pred_train, _ = svm.predict(X_train_3, threshold=threshold)
    train_acc = accuracy(Y_pred_train, Y_train_3)
    print(f"Train Accuracy : {train_acc}")

    Y_pred_test, _ = svm.predict(X_test_3, threshold=threshold)
    test_acc = accuracy(Y_pred_test, Y_test_3)
    print(f"Test Accuracy : {test_acc}")

    alpha_sv, X_sv = svm.get_coeff_sv(threshold=1e-6)
    print(f"Number of SV : {len(alpha_sv)}")

    with open('./data/y_pred_g.pickle', 'wb') as f:
        pickle.dump(Y_pred_test, f)

elif part_num == 'c':

    """
    LIBSVM
    """
    print("-- Question 2b. iii", end='\n\n')

    """
    Linear SVM
    """
    print("-Linear SVM", end='\n\n')
    prob  = svm_problem(Y_train_3.reshape(-1), X_train_3)
    param = svm_parameter('-t 0 -c 1 -q')

    start = time.time()
    m = svm_train(prob, param)
    end = time.time()
    print(f"Training time : {end-start} secs")

    p_labels_t, p_acc_t, p_vals_t = svm_predict(Y_train_3.reshape(-1),
                                          X_train_3, m, options='-q')
    print(f"Train Accuracy : {p_acc_t[0]}")
    p_labels, p_acc, p_vals = svm_predict(Y_test_3.reshape(-1),
                                          X_test_3, m, options='-q')
    print(f"Test Accuracy : {p_acc[0]}")

    print(f"Number of SV : {m.get_nr_sv()}", end='\n\n')


    with open('./data/y_pred_libl.pickle', 'wb') as f:
        pickle.dump(p_labels, f)


    """
    Gaussian SVM
    """
    print("- Gaussian SVM", end='\n\n')
    prob  = svm_problem(Y_train_3.reshape(-1), X_train_3)
    param = svm_parameter('-t 2 -c 1 -g 0.05 -q')

    start = time.time()
    m = svm_train(prob, param)
    end = time.time()
    print(f"Training time : {end-start} secs")


    p_labels_t, p_acc_t, p_vals_t = svm_predict(Y_train_3.reshape(-1),
                                          X_train_3, m, options='-q')
    print(f"Train Accuracy : {p_acc_t[0]}")
    p_labels, p_acc, p_vals = svm_predict(Y_test_3.reshape(-1),
                                          X_test_3, m, options='-q')
    print(f"Test Accuracy : {p_acc[0]}")
    print(f"Number of SV : {m.get_nr_sv()}")


    with open('./data/y_pred_libg.pickle', 'wb') as f:
        pickle.dump(p_labels, f)

else:
    print("Invalid part number")

