from binary_svm import *

"""
class that performs one vs one multiclass SVM classification with
gaussian kernel
"""
class OvO_GSVM:

    def __init__(self):
        self.ovo_svms = {}
        self.classes = None

    def train(self, X_train, Y_train, X_val, Y_val, C=1.0, gamma=0.05,
              threshold=1e-6):
        self.classes = np.unique(Y_train)

        for i in range(len(self.classes)):
            for j in range(i+1, len(self.classes)):

                #data collection
                X_train_ij, Y_train_ij = two_class_data(X_train,
                                                        Y_train,
                                                        self.classes[i],
                                                        self.classes[j])
                X_val_ij, Y_val_ij = two_class_data(X_val, Y_val,
                                                    self.classes[i],
                                                    self.classes[j])
                #training
                svm = GaussianSVM()
                svm.train(X_train_ij, Y_train_ij, C, gamma)
                self.ovo_svms[(self.classes[i], self.classes[j])] = svm

                #accuracy
                Y_pred_ij, _ = svm.predict(X_val_ij, threshold=threshold)
                acc = accuracy(Y_pred_ij, Y_val_ij)
                print(f"{self.classes[i]}-{self.classes[j]} Accuracy : {acc}")


    def ovo_svm_prediction(self, Y_pred_ij, Y_score_ij):
        ovo_pred = []
        for i in range(Y_pred_ij.shape[0]):
            count = np.bincount(Y_pred_ij[i])
            max_count = np.max(count)

            labels = np.where(count == max_count)[0]
            label_score = Y_score_ij[i][labels]

            label = labels[np.argmax(label_score)]
            ovo_pred.append(label)
        return np.array(ovo_pred).reshape(-1, 1)


    def predict(self, X_test, threshold=1e-6):
        Y_pred_ijs = []
        Y_score_ijs = []
        for i in range(len(self.classes)):
            for j in range(i+1, len(self.classes)):
                svm = self.ovo_svms[(self.classes[i], self.classes[j])]

                Y_pred_ij, Y_score_ij = svm.predict(X_test, threshold)
                Y_pred_ij = np.where( Y_pred_ij == 1,
                                     self.classes[i], self.classes[j])

                Y_pred_ijs.append(Y_pred_ij)
                Y_score_ijs.append(np.abs(Y_score_ij))

        return self.ovo_svm_prediction(np.hstack(Y_pred_ijs),
                                  np.hstack(Y_score_ijs))


"""
K fold cross validation
"""
def Kfold_validation_C(X_train, Y_train, C, K=5):
    accuracies = {}
    batch_size = int(np.ceil(X_train.shape[0]/K))

    for c in C:
        print(f"Parameter value : {c}")

        fold_accuracy = []
        for k in range(K):
            val_start = k*batch_size
            val_end = (k+1)*batch_size

            X_val = X_train[val_start:val_end]
            y_val = Y_train[val_start:val_end]

            X_train_fold = np.vstack([X_train[:val_start],
                                      X_train[val_end:]])
            y_train_fold = np.vstack([Y_train[:val_start],
                                      Y_train[val_end:]])

            prob  = svm_problem(y_train_fold.reshape(-1),
                                X_train_fold)
            param = svm_parameter(f'-t 2 -c {c} -g 0.05 -q')
            m = svm_train(prob, param)
            p_labels, p_acc, p_vals = svm_predict(y_val.reshape(-1),
                                                  X_val, m,
                                                  options='-q')
            fold_accuracy.append(p_acc[0])

        accuracies[c] = fold_accuracy
    return accuracies



"""
Test accuracy on Kfold
"""
def test_accuracy_C(X_train, Y_train, X_test, Y_test, C):
    accuracies = {}
    prob  = svm_problem(Y_train.reshape(-1), X_train, isKernel=True)
    for c in C:
        print(f"Parameter value : {c}")
        param = svm_parameter(f'-t 2 -c {c} -g 0.05 -q')
        m = svm_train(prob, param)
        p_labels, p_acc, p_vals = svm_predict(Y_test.reshape(-1),
                                              X_test, m)
        accuracies[c] = p_acc[0]
    return accuracies

def confusion_matrix_and_visualize(X, y, file_name, message):
    classes = np.unique(y)

    if not os.path.exists(file_name) or not os.path.exists(file_name):
        print("ERROR: run previous parts, 2a i, ii, iii and 2b i, ii")
        exit(0)

    print(message, end='\n\n')
    with open(file_name, 'rb') as f:
        Y_pred = pickle.load(f)
    Y_pred = np.array(Y_pred, dtype=int).reshape(-1, 1)

    #Confusion matrix
    conf_df = create_confusion_matrix(Y_pred, y, classes)
    print("Confusion Matrix")
    display(conf_df)

    #Most misclassified
    conf_matrix = conf_df.to_numpy()
    print(f"Most misclassified : {most_misclassified(conf_matrix)}")

    visualize_misclassified(X, y, Y_pred, num=10, cols=5)

