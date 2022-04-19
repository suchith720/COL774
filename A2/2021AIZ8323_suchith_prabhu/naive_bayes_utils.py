import os
import re
import nltk
import time
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

import argparse
from IPython.display import display

plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

"""
For creating term document matrix
"""
def countVectorizer(docs, vocabulary=None):
    indptr = [0]
    indices = []
    term_freq = []

    if vocabulary is None:
        fixed_vocabulary = False
        vocabulary = {}
    else:
        fixed_vocabulary = True

    for doc in docs:
        for term in doc:
            if not fixed_vocabulary or term in vocabulary:
                index = vocabulary.setdefault(term, len(vocabulary))
                indices.append(index)
                term_freq.append(1)
        indptr.append(len(indices))

    term_doc_matrix = csr_matrix((term_freq, indices, indptr),
                                 shape=(len(docs), len(vocabulary)),
                                 dtype=int)

    return vocabulary, term_doc_matrix


"""
convert to lowercase and remove punctuations and characters
"""
def text_cleaning(data):
    clean_data = []
    for i, doc in enumerate(data):
        clean_doc = re.sub(r'[^\w\s]', ' ', str(doc).lower().strip())
        clean_doc = re.sub(r' [\w\s] ', ' ', clean_doc).strip()
        clean_data.append(clean_doc)
    return clean_data


"""
Function for tokenization
"""
nltk.download('punkt', quiet=True)
def tokenize(data):
    return word_tokenize(data)

def text_processor_1(data):
    data_tokens = []
    for doc in data:
        data_tokens.append(tokenize(doc))
    return data_tokens


"""
to create term document matrix
"""
def extract_feature(X_train_input, X_test_input, extract_func, file):

    if not os.path.exists(file):
        X_train_tokens = extract_func(X_train_input)
        vocab, X_train = countVectorizer(X_train_tokens)

        X_test_tokens = extract_func(X_test_input)
        vocab, X_test = countVectorizer(X_test_tokens, vocab)

        with open(file, "wb") as f:
            pickle.dump([X_train_tokens, X_test_tokens], f)
    else:
        with open(file, "rb") as f:
            X_train_tokens, X_test_tokens = pickle.load(f)

        vocab, X_train = countVectorizer(X_train_tokens)
        vocab, X_test = countVectorizer(X_test_tokens, vocab)

    return X_train, X_test, X_train_tokens, X_test_tokens, vocab


"""
Training code
"""
def train_mnb(X_train, Y_train, classes):

    phi = []
    theta = []

    for k in classes:
        phi_k = np.mean(Y_train == k)
        phi.append(phi_k)

        term_k = X_train[Y_train == k].sum(axis=0) + 1
        theta_k = np.squeeze(np.asarray(term_k))/term_k.sum()
        theta.append(theta_k)

    return np.array(phi), np.array(theta)


"""
Model prediction code
"""
def predict_mnb(phi, theta, X_test, classes):
    log_theta = np.log(theta).T
    Y_score = X_test@log_theta + phi
    Y_pred = np.argmax(Y_score, axis=1)

    Y_pred = np.array([classes[y] for y in Y_pred])
    return Y_pred


"""
Model Accuracy
"""
def accuracy(Y_pred, Y):
    return np.mean(Y_pred == Y)


"""
Confusion matrix
"""
def create_confusion_matrix(Y_pred, Y, classes):
    confusion_matrix = []

    for class_1 in classes:
        Y_class = Y_pred[Y==class_1]
        class_count = []
        for class_2 in classes:
            class_count.append(np.sum(Y_class == class_2))
        confusion_matrix.append(class_count)

    return pd.DataFrame(confusion_matrix,
                        index=pd.Index(classes, name='Actual :'),
                        columns=pd.Index(classes, name='Predicted :') )


"""
computes f1 score
"""
def F1_score(precision, recall):
    n = 2*precision*recall
    d = precision + recall
    return n/d



"""
computes f1 score from confusion matrix
"""
def compute_F1_from_confusion(conf_df):
    conf_matrix = conf_df.to_numpy()
    f1_scores = []

    for c in range(len(conf_matrix)):
        fp_tp = conf_matrix[:,c].sum()
        if fp_tp:
            precision = conf_matrix[c, c]/fp_tp
        else:
            precision = 1

        recall = conf_matrix[c, c]/conf_matrix[c, :].sum()
        f1 = F1_score(precision, recall)
        f1_scores.append(f1)

    return pd.DataFrame(f1_scores,
                        index=conf_df.columns.to_list(),
                        columns=['F1 score'])



"""
performs training and prediction
"""
def train_predict(X_train, Y_train, X_test, Y_test, classes):
    # Training
    phi, theta = train_mnb(X_train, Y_train, classes)
    model_param = (phi, theta)

    # Prediction on training data
    Y_pred_train = predict_mnb(phi, theta, X_train, classes)
    train_acc = accuracy(Y_pred_train, Y_train)
    train_conf_df = create_confusion_matrix(Y_pred_train,
                                            Y_train, classes)
    train_f1_df = compute_F1_from_confusion(train_conf_df).T
    train_macro_f1 = train_f1_df.mean(axis=1).to_numpy()[0]
    train_predictions = (Y_pred_train, train_acc, train_conf_df,
                         train_f1_df, train_macro_f1)

    # Prediction on testing data
    Y_pred = predict_mnb(phi, theta, X_test, classes)
    test_acc = accuracy(Y_pred, Y_test)
    conf_df = create_confusion_matrix(Y_pred, Y_test, classes)
    f1_df = compute_F1_from_confusion(conf_df).T
    macro_f1 = f1_df.mean(axis=1).to_numpy()[0]
    test_predictions = (Y_pred, test_acc, conf_df, f1_df, macro_f1)

    return model_param, train_predictions, test_predictions


def load_tokens(file):
    if not os.path.exists(file):
        print('ERROR: Run previous parts')
        exit(0)

    with open(file, 'rb') as f:
        X_train_tokens, X_test_tokens = pickle.load(f)

    return X_train_tokens, X_test_tokens


def print_prediction_stats(train_predictions, test_predictions):

    # Train predictions
    print("TRAINING STATS", end='\n\n')
    Y_pred_train, train_acc, train_conf_df,train_f1_df, train_macro_f1 = train_predictions
    print(f"Train Accuracy : {train_acc}", end='\n\n')

    # Test predictions
    print("TESTING STATS", end='\n\n')
    Y_pred, test_acc, conf_df, f1_df, macro_f1 = test_predictions

    print(f"Test Accuracy : {test_acc}", end='\n\n')
    print("Confusion Matrix :")
    display(conf_df)
    print()
    display(f1_df)
    print()
    print(f"Macro f1 : {macro_f1}")




"""
Stemming and stopword removal
"""
def stemming(data):
    stems = [PorterStemmer().stem(token) for token in data]
    return stems

def remove_stop_words(data, stopwords):
    tokens = [word for word in data if word not in stopwords]
    return tokens

def text_processor_2(data, stopwords):
    data_tokens = []
    for i, doc in enumerate(data):
        if i%1000 == 0:
            print(f"Processing : {i}", end='\r', flush=True)
        tokens = remove_stop_words(doc, stopwords)
        tokens = stemming(tokens)
        data_tokens.append(tokens)
    return data_tokens



"""
Creates bi-gram features
"""
def concat_bigram(data):
    data_tokens = []
    for doc in data:
        doc_bi_gram = []
        for k in range(1, len(doc)):
            doc_bi_gram.append(f"{doc[k-1]} {doc[k]}")
        data_tokens.append(doc+doc_bi_gram)
    return data_tokens

"""
Creates tri-gram features
"""
def concat_trigram(data):
    data_tokens = []
    for doc in data:
        doc_tri_gram = []
        for k in range(2, len(doc)):
            doc_tri_gram.append(f"{doc[k-2]} {doc[k-1]} {doc[k]}")
        data_tokens.append(doc+doc_tri_gram)
    return data_tokens
