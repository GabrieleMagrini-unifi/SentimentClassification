import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn import tree
import nltk
import re


data = pd.read_csv("data/data_review_balanced.tsv", delimiter="\t")

print("Dataset: ", data)

rece = open("data/receFIFA21.txt")
string_without_line_breaks = ""
for line in rece:
  stripped_line = line.rstrip()
  string_without_line_breaks += stripped_line
rece = [string_without_line_breaks]

print("Review: ", rece)


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


vectorizer = CountVectorizer(binary=True, min_df=4)       #ngram_range= (1,2) per unigrams+bigrams , =(2,2) per soli bigrams. Con binary = False, Frequency.


X_train, X_test, Y_train, Y_test, n = simple_split(data_review_copy, data_sentiment_copy, len(data))


X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("# of Features: ", len(vectorizer.vocabulary_))

X_try = vectorizer.transform(rece)

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)

print("Test score Perceptron: ", perceptron.score(X_test, Y_test))

tree = tree.DecisionTreeClassifier()
tree.fit(X_train, Y_train)

print("Test score Tree: ", tree.score(X_test, Y_test))

print("Review sentiment based on perceptron :", perceptron.predict(X_try))
print("Review sentiment based on tree :",tree.predict(X_try))
