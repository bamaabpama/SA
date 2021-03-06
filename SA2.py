# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 12:16:11 2018

@author: DELL
"""

import pymysql as mdb
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,precision_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('E:/Digitalent Big Data/tugas pak ayok/cleandata-label.csv', encoding='utf-8')
df.head()

a=df.label.value_counts()
print(a)
a=df.dropna()

from sklearn.model_selection import train_test_split

X = a.berita
y = a.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#from sklearn.feature_extraction.text import CountVectorizer
#
vect = TfidfVectorizer()
#
X_train_vect = vect.fit_transform(X_train)

sm = SMOTE()

X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)
unique, counts = np.unique(y_train_res, return_counts=True)
print(list(zip(unique, counts)))


#
#nb = MultinomialNB()
#
#nb.fit(X_train_res, y_train_res)
#
#print(nb.score(X_train_res, y_train_res))
#
#X_test_vect = vect.transform(X_test)
#
#y_pred = nb.predict(X_test_vect)


#svm = LinearSVC()
#
#svm.fit(X_train_res, y_train_res)
#
#print(svm.score(X_train_res, y_train_res))
#
#X_test_vect = vect.transform(X_test)
#
#y_pred = svm.predict(X_test_vect)


#reg = LogisticRegression()
#
#reg.fit(X_train_res, y_train_res)
#
#print(reg.score(X_train_res, y_train_res))
#
#X_test_vect = vect.transform(X_test)
#
#y_pred = reg.predict(X_test_vect)



#forest = RandomForestClassifier()
#
#forest.fit(X_train_res, y_train_res)
#
#print(forest.score(X_train_res, y_train_res))
#
#X_test_vect = vect.transform(X_test)
#
#y_pred = forest.predict(X_test_vect)
#
#
#
#print(precision_score(y_test, y_pred, average='macro')) 
#print(accuracy_score(y_test, y_pred))
#print(confusion_matrix(y_test,y_pred))

X_test_vect = vect.transform(X_test)

models = [
    MultinomialNB(),
    LogisticRegression(),
    LinearSVC(),
    RandomForestClassifier(),
]

for model in models:
    slashes = '-' * 30
    print(f"\n{slashes}\n{model.__class__.__name__}\n{slashes}")
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_vect)
        
    print(precision_score(y_test, y_pred, average='macro')) 
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test,y_pred))
        

