from naive_bayes_utils import *

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

train_data = pd.read_json(train_data_path, lines=True)
X_train_text = train_data["reviewText"].to_numpy()
Y_train = train_data["overall"].to_numpy()

test_data = pd.read_json(test_data_path, lines=True)
X_test_text = test_data["reviewText"].to_numpy()
Y_test = test_data["overall"].to_numpy()

classes = np.unique(Y_train)


#For storing output features
os.makedirs("./data/", exist_ok=True)



if part_num == 'a':
    """
    Question 1a : training naive bayes
    --------------------------------
    """
    print("-- Question 1a")

    """
    Text cleaning
    """
    X_train_clean = text_cleaning(X_train_text)
    X_test_clean = text_cleaning(X_test_text)


    """
    tokenzation and term document matrix creating
    """
    X_train, X_test, X_train_tokens, X_test_tokens, vocab = extract_feature(X_train_clean, X_test_clean, text_processor_1, "./data/tokens.pickle")

    print(f"vocabulary size : {len(vocab)}", end="\n\n")


    """
    Traininig and Predictions
    """
    model_param, train_predictions, test_predictions = train_predict(
        X_train, Y_train, X_test, Y_test, classes)

    with open("./data/mnb_params.pickle", 'wb') as f:
        pickle.dump(model_param, f)

    print_prediction_stats(train_predictions, test_predictions)

    #saving for part 1c
    _, _, conf_df, _, _ = test_predictions
    with open('./data/1a_conf_matrix.pickle', 'wb') as f:
        pickle.dump(conf_df, f)


elif part_num == 'b':
    """
    Question 1b : comparing accuracy
    --------------------------------
    """
    print("-- Question 1b")


    """
    Random accuracy
    """
    Y_pred = np.random.randint(1, 6, Y_test.shape)
    random_acc = accuracy(Y_pred, Y_test)
    print(f"Random Accuracy : {random_acc}")

    """
    Majority accuracy
    """
    if not os.path.exists("./data/mnb_params.pickle"):
        print("ERROR: Run previous parts")
        exit(0)

    with open("./data/mnb_params.pickle", 'rb') as f:
        phi, theta = pickle.load(f)

    maxClass = classes[np.argmax(phi)]
    m_acc = accuracy(maxClass, Y_test)
    print(f"Majority Accuracy : {m_acc}")


elif part_num == 'c':
    """
    Question 1c : Confusion matrix
    ------------------------------
    """
    print("-- Question 1c")
    if not os.path.exists('./data/1a_conf_matrix.pickle'):
        print("ERROR: Run previous parts")
        exit(0)

    with open('./data/1a_conf_matrix.pickle', 'rb') as f:
        conf_df = pickle.load(f)
        display(conf_df)

elif part_num == 'd':
    """
    Question 1d : Stemming and Stopword removal
    ------------------------------
    """
    print("-- Question 1d")

    X_train_tokens, X_test_tokens = load_tokens('./data/tokens.pickle')


    """
    term-doc matrix creation
    """
    stopwords = nltk.corpus.stopwords.words('english')
    extract_func = lambda X : text_processor_2(X, stopwords)

    X_train, X_test, X_train_stems, X_test_stems, vocab = extract_feature(
        X_train_tokens, X_test_tokens, extract_func, "./data/stems.pickle")

    print(f"vocabulary size : {len(vocab)}", end='\n\n')


    """
    Traininig and Prediction
    """
    model_param, train_predictions, test_predictions = train_predict(
        X_train, Y_train, X_test, Y_test, classes)

    print_prediction_stats(train_predictions, test_predictions)

elif part_num == 'e':
    """
    Question 1e : feature engineering
    ---------------------------------
    """
    print("-- Question 1e")
    X_train_stems, X_test_stems = load_tokens('./data/stems.pickle')

    """
    Bi gram model
    """
    print("- Bi-gram model", end='\n\n')
    X_train, X_test, X_train_bigrams, X_test_bigrams, vocab = extract_feature(
        X_train_stems, X_test_stems, concat_bigram, "./data/bigram.pickle")

    print(f"vocabulary size : {len(vocab)}", end='\n\n')

    model_param, train_predictions, test_predictions = train_predict(
        X_train, Y_train, X_test, Y_test, classes)
    print_prediction_stats(train_predictions, test_predictions)


    """
    Trigram model
    """
    print("- Tri-gram model", end='\n\n')
    X_train, X_test, X_train_trigrams, X_test_trigrams, vocab = extract_feature(
        X_train_stems, X_test_stems, concat_trigram, "./data/trigram.pickle")

    print(f"vocabulary size : {len(vocab)}", end='\n\n')
    model_param, train_predictions, test_predictions = train_predict(
        X_train, Y_train, X_test, Y_test, classes)
    print_prediction_stats(train_predictions, test_predictions)

elif part_num == 'g':
    """
    Question 1g : Summary
    ---------------------------------
    """
    print("-- Question 1g")

    """
    Combining summary and review text
    """
    X_train_sum = train_data["summary"].to_numpy()
    X_test_sum = test_data["summary"].to_numpy()

    X_train_comb = X_train_text + X_train_sum
    X_test_comb = X_test_text + X_test_sum

    print("Text cleaning")
    X_train_clean = text_cleaning(X_train_comb)
    X_test_clean = text_cleaning(X_test_comb)

    X_train_sum_tokens = text_processor_1(X_train_clean)
    vocab, X_train_sum = countVectorizer(X_train_sum_tokens)
    X_test_sum_tokens = text_processor_1(X_test_clean)
    vocab, X_test_sum = countVectorizer(X_test_sum_tokens, vocab)

    stopwords = nltk.corpus.stopwords.words('english')
    X_train_sum_stems = text_processor_2(X_train_sum_tokens, stopwords)
    vocab, X_train = countVectorizer(X_train_sum_stems)
    X_test_sum_stems = text_processor_2(X_test_sum_tokens, stopwords)
    vocab, X_test = countVectorizer(X_test_sum_stems, vocab)

    print(f"vocabulary size : {len(vocab)}", end='\n\n')
    model_param, train_predictions, test_predictions = train_predict(
        X_train, Y_train, X_test, Y_test, classes)
    print_prediction_stats(train_predictions, test_predictions)


else:
    print("ERROR: Invalid part number")





