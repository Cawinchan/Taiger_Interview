import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


import nltk
nltk.download('punkt')

train = pd.read_csv("~/Downloads/train.csv",  sep='\t')
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100
pd.set_option('max_colwidth', 100)
print(train.shape)

train.fillna('',inplace=True)
print(train.head())
print(train.columns)


train['review'] = train.loc[:,'review':].astype(str).apply(lambda x: ''.join(x),axis=1)

train.drop(train.loc[:,'Unnamed: 3':],inplace=True,axis=1)

print(train.head())

zero = 0
one = 0
for row in train['sentiment']:
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

train['sentiment'].describe()


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())



import re


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


#print(train['review'].head(10))

for row in train['review']:
    cleanfromhtml = cleanhtml(row)
    cleanfromnumbers_html = re.sub("\d+", "", cleanfromhtml)
    cleanfromnumbers_html_punc = re.sub(r'[^\w\s]', '', cleanfromnumbers_html)
    train.loc[train['review'] == str(row), ['review']] = cleanfromnumbers_html_punc

#print(train.head(10))



x_train,x_test,y_train,y_test = train_test_split(train['review'].astype(str),train['sentiment'].astype(int),test_size=0.33)

#print(x_train.head(),y_train.head())


from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import tensorflow

from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras

import tensorflow as tf
import os

tf.reset_default_graph()


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
    early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=0, mode='auto')

    sparse_model.fit(vector, sentiment, epochs=10,
                     callbacks=[early_stop_callback, ])
    sparse_model.evaluate(x_test, y_test)
    validation = sparse_model.predict_classes(x_test)
    print(confusion_matrix(y_test, validation))
    print(classification_report(y_test, validation))
    return sparse_model


def tf_idf_vectoriser(sentences):
    vect_word = TfidfVectorizer(stop_words='english')
    sparse_matrix_word = vect_word.fit_transform(sentences)
    X_test = vect_word.transform(x_test)
    return sparse_matrix_word, X_test


def count_vectoriser(sentences):
    cvec = CountVectorizer(stop_words='english')
    sparse_matrix_count = cvec.fit_transform(sentences)
    X_test = cvec.transform(x_test)
    return sparse_matrix_count, X_test


def Hashing_Vectorizer(sentences):
    hv = HashingVectorizer(stop_words='english')
    sparse_matrix_count = hv.fit_transform(sentences)
    X_test = hv.transform(x_test)
    return sparse_matrix_count, X_test


def build_logistic_regression_model(vector, sentiment, x_test, y_test):
    lr = LogisticRegression()
    lr.fit(vector, sentiment)
    y_pred = lr.predict(X_test)
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


def build_linear_SVC_model(vector, sentiment, x_test, y_test):
    lsvc = LinearSVC()
    lsvc.fit(vector, sentiment)
    y_pred = lsvc.predict(x_test)
    confusion_matrix2 = confusion_matrix(y_test, y_pred)
    print(confusion_matrix2)
    print(classification_report(y_test, y_pred))


#sparse1, X_test = tf_idf_vectoriser(x_train)
#rnn_network(sparse1, y_train, X_test, y_test)

#sparse2, X_test = tf_idf_vectoriser(x_train)
#build_logistic_regression_model(sparse2, y_train, X_test, y_test)
#build_linear_SVC_model(sparse2, y_train, X_test, y_test)

#sparse3, X_test = count_vectoriser(x_train)
#build_logistic_regression_model(sparse3, y_train, X_test, y_test)
#build_linear_SVC_model(sparse3, y_train, X_test, y_test)

#sparse4, X_test = Hashing_Vectorizer(x_train)
#build_logistic_regression_model(sparse4, y_train, X_test, y_test)
#build_linear_SVC_model(sparse4, y_train, X_test, y_test)



from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer

def build_MultinomialNB(vector, sentiment, x_test, y_test):
    nb = MultinomialNB().fit(vector,sentiment)
    y_pred = nb.predict(x_test)
    confusion_matrix2 = confusion_matrix(y_test, y_pred)
    print(confusion_matrix2)
    print(classification_report(y_test, y_pred))



cv = CountVectorizer()
x_train_transformed = cv.fit_transform(x_train)
X_test = cv.fit(x_test)



td_trans = TfidfTransformer()
sparse5 = td_trans.fit_transform(x_train_transformed)
X_test = td_trans.fit(X_test)


build_MultinomialNB(sparse5.reshape(-1,1),y_train,X_test,y_test)

