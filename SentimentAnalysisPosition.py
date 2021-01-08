import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn import tree
import nltk
import re


data = pd.read_csv("data/labeledTrainData.tsv", delimiter="\t")


def simple_split(data, y, lenght, split_mark=0.7):
    if 0. < split_mark < 1.0:
        n = int(split_mark * lenght)
    else:
        n = int(split_mark)
    X_train = data[:n].copy()
    X_test = data[n:].copy()
    Y_train = y[:n].copy()
    Y_test = y[n:].copy()
    return X_train, X_test, Y_train, Y_test, n


data_review_copy = data.review.copy()
data_sentiment_copy = data.sentiment.copy()



vectorizer = CountVectorizer(binary=True, min_df=4)                                         # Con binary = False, Frequency.


X_train, X_test, Y_train, Y_test, n = simple_split(data_review_copy, data_sentiment_copy, len(data))

print(X_test[n])

for i in range(len(X_train)-1):
    X_train[i] = re.findall(r'\w+', X_train[i])
    X_train_ult = []
    for j in range(0, len(X_train[i])-1):
        if j < len(X_train[i])//4:
            X_train_ult.append(X_train[i][j] + '_firstQuarter')
        elif len(X_train[i])//4 <= j < len(X_train[i])-len(X_train[i])//4:
            X_train_ult.append(X_train[i][j] + '_middleHalf')
        else:
            X_train_ult.append(X_train[i][j] + '_lastQuarter')
    X_train[i] = X_train_ult

for i in range(len(X_train)-1):
    X_train[i] = str(X_train[i])

print(X_train[10])


for i in range(n, len(data)):
    X_test[i] = re.findall(r'\w+', X_test[i])
    X_test_ult = []
    for j in range(0, len(X_test[i]) - 1):
        if j < len(X_test[i]) // 4:
            X_test_ult.append(X_test[i][j] + '_firstQuarter')
        elif len(X_test[i]) // 4 <= j < len(X_test[i]) - len(X_test[i]) // 4:
            X_test_ult.append(X_test[i][j] + '_middleHalf')
        else:
            X_test_ult.append(X_test[i][j] + '_lastQuarter')
    X_test[i] = X_test_ult

for i in range(n, len(data)):
    X_test[i] = str(X_test[i])

print(X_test[n+1])

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print(len(vectorizer.vocabulary_))

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
print("Test score Perceptron: ", perceptron.score(X_test, Y_test))

tree = tree.DecisionTreeClassifier()
tree.fit(X_train, Y_train)
print("Test score Tree: ", tree.score(X_test, Y_test))
