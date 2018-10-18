import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer

from scipy import sparse

from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras

import re

import matplotlib.pyplot as plt




def General_sentiment_viewer(column):
    zero = 0                                           # Counts the number of 0 and 1s
    one = 0
    for row in column:
        if row == 0:
            zero += 1
            continue
        if row == 1:
            one += 1
            continue
        else:
            print(row,"missing response")
    print("zero count",zero)
    print("one count",one)
    print("total is", zero+one)

    column.describe()
    return




def cleantext(df, raw_html, troubleshoot=False):
    cleanr = re.compile('<.*?>')

    if troubleshoot:
        print(raw_html.head(10))

    for row in raw_html:
        cleanfromhtml = re.sub(cleanr, '', row)
        cleanfromnumbers_html = re.sub("\d+", "", cleanfromhtml)
        cleanfromnumbers_html_punc = re.sub(r'[^\w\s]', '', cleanfromnumbers_html)
        df.loc[raw_html == str(row), ['review']] = cleanfromnumbers_html_punc

    if troubleshoot:
        print(raw_html.head(10))
    return


def rnn_network(vector, sentiment, x_test, y_test):
    sparse_model = Sequential()
    sparse_model.add(Dense(128, input_shape=(vector.shape[1],)))
    sparse_model.add(Dropout(0.2))
    sparse_model.add(Dense(100))
    sparse_model.add(Dropout(0.2))
    sparse_model.add(Dense(50))
    sparse_model.add(Dropout(0.2))
    sparse_model.add(Dense(10))
    sparse_model.add(Dropout(0.2))
    sparse_model.add(Dense(1, activation='sigmoid'))
    sparse_model.compile(optimizer='rmsprop',
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
    early_stop_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=4, verbose=0, mode='auto')

    sparse_model.fit(vector, sentiment, epochs=20,
                     callbacks=[early_stop_callback, ])
    sparse_model.evaluate(x_test, y_test)
    validation = sparse_model.predict_classes(x_test)
    print(confusion_matrix(y_test, validation))
    print(classification_report(y_test, validation))
    return sparse_model


def tf_idf_vectoriser(sentences):
    vect_word = TfidfVectorizer(stop_words='english')
    sparse_matrix_word = vect_word.fit_transform(sentences)
    return sparse_matrix_word


def count_vectoriser(sentences):
    cvec = CountVectorizer(stop_words='english')
    sparse_matrix_count = cvec.fit_transform(sentences)
    return sparse_matrix_count


def hashing_Vectorizer(sentences):
    hv = HashingVectorizer(stop_words='english')
    sparse_matrix_count = hv.fit_transform(sentences)
    return sparse_matrix_count


def build_logistic_regression_model(vector, sentiment, x_test, y_test):
    lr = LogisticRegression()
    lr.fit(vector, sentiment)
    y_pred = lr.predict(x_test)
    confusion_matrix2 = confusion_matrix(y_test, y_pred)
    print(confusion_matrix2)
    print(classification_report(y_test, y_pred))
    logit_roc_auc = roc_auc_score(y_test, lr.predict(x_test))
    fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(x_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    return lr


def build_linear_SVC_model(vector, sentiment, x_test, y_test):
    lsvc = LinearSVC()
    lsvc.fit(vector, sentiment)
    y_pred = lsvc.predict(x_test)
    confusion_matrix2 = confusion_matrix(y_test, y_pred)
    print(confusion_matrix2)
    print(classification_report(y_test, y_pred))
    return lsvc


def run_model(vectoriser, model, description, counter):
    spar = vectoriser(description)
    x_train, x_test, y_train, y_test = train_test_split(spar, counter, test_size=0.33)
    model(x_train, y_train, x_test, y_test)
    return spar



def build_MultinomialNB(vector, sentiment, x_test, y_test):
    nb = MultinomialNB().fit(vector,sentiment)
    y_pred = nb.predict(x_test)
    confusion_matrix2 = confusion_matrix(y_test, y_pred)
    print(confusion_matrix2)
    print(classification_report(y_test, y_pred))
    return nb

def build_KNeighborsClassifier(description, counter, x_test, y_test):
    kn = KNeighborsClassifier().fit(description,counter)
    y_pred = kn.predict(x_test)
    confusion_matrix2 = confusion_matrix(y_test, y_pred)
    print(confusion_matrix2)
    print(classification_report(y_test, y_pred))
    return kn

def build_RandomForestClassifier(description, counter, x_test, y_test):
    rf = RandomForestClassifier().fit(description,counter)
    y_pred = rf.predict(x_test)
    confusion_matrix2 = confusion_matrix(y_test, y_pred)
    print(confusion_matrix2)
    print(classification_report(y_test, y_pred))
    return rf


if __name__ == "__main__":
    train = pd.read_csv("~/Downloads/train.csv", sep='\t')  # reformating data for pandas
    pd.options.display.max_rows = 100
    pd.options.display.max_columns = 100
    pd.set_option('max_colwidth', 100)
    print(train.shape)


    train.fillna('', inplace=True)
    print(train.head())
    print(train.columns)

    train['review'] = train.loc[:, 'review':].astype(str).apply(lambda x: ''.join(x),
                                                                axis=1)  # remove spaces and join all the columns used for the length for the reviews
    train.drop(train.loc[:, 'Unnamed: 3':], inplace=True, axis=1)  # remove the empty columns
    cleantext(train, train['review'], troubleshoot=False)
    print(train.head())

    sentiment = train['sentiment']
    review = train['review']

    General_sentiment_viewer(sentiment)


    description = train['review'].tolist()
    counter = train['sentiment'].tolist()
    counter = np.array(counter)
    description = np.array(description)

    #Exploration of rnn, lr and svc models
    rnn = run_model(tf_idf_vectoriser,rnn_network,description,counter)
    lr = run_model(tf_idf_vectoriser,build_logistic_regression_model,description,counter)
    svc= run_model(tf_idf_vectoriser,build_linear_SVC_model,description,counter)

    # Exploration of CountVectorizer, tftransformer with Naive Bayes
    cv = CountVectorizer()
    x_train_transformed = cv.fit_transform(description)


    td_trans = TfidfTransformer()
    sparse5 = td_trans.fit_transform(x_train_transformed)

    train_x, test_x, train_y, test_y = train_test_split(sparse5, counter, test_size=0.33)

    nb = build_MultinomialNB(train_x,train_y,test_x,test_y)


    #Exploration of count Vectoriser with tf-idf and logistic regression
    sparse2 = tf_idf_vectoriser(description)
    train_x, test_x, train_y, test_y = train_test_split(sparse2, counter, test_size=0.33)

    lr2 = build_logistic_regression_model(train_x,train_y, test_x,test_y)

    kn = run_model(tf_idf_vectoriser,build_KNeighborsClassifier,description,counter)

    rf = run_model(tf_idf_vectoriser,build_RandomForestClassifier,description,counter)

    # Exploration of deep and Wide Model With sparse matrixes from lr and svc with the meta classfier of RNN (most consistent 80% results
    sparse_matrix_combined = sparse.hstack([lr, svc])
    print(sparse_matrix_combined.shape)

    train_x, test_x, train_y, test_y = train_test_split(sparse_matrix_combined, counter, test_size=0.33)

    rnn_network(train_x,train_y,test_x,test_y)



    # Different types of pre-processing methods are important as there are many ways to convert or translate words into machine langauge,
    # Additionally, as language is a multi-dimensional subject there are many ways in which it can be interpreted by a computer.
    # A language can be broken down into its basic structure, a phrase of words or it can be look at by each individual word itself, as such
    # find the right weightage of each is important in processing it to machine langugage. It is for this reason that I have tried to use a
    # deep and wide model with the RNN looking for the structure of the words while the logistic regression and SVC looks for each individual word.
    # However, there is still much left unexplored in the area of lauguage sematics with the regards to machine learning. As such, it still poses as
    # a problem that stumps even the greatest minds in this era. I have grown quite fond of linguistics due to my literature background. As such this
    # was a fun and rewarding project for me. In this regard, this ties in with the business strategy of taiger and I hope my knowledge can not only
    # grow as a programmer but to also help improve Taiger.