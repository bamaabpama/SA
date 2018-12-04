# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 12:16:11 2018

@author: DELL
"""

import pymysql as mdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,average
from imblearn.over_sampling import SMOTE
import pandas as pd

#con = mdb.connect(host='localhost',user='root',password='',database='gdelt')
# 
## inisialisasi cursor object method
#cur = con.cursor()
# 
## eksekusi query
#cur.execute("SELECT * FROM cleanlabel")
# 
## ambil semua data dari query
#rows = cur.fetchall()
#
#i = 0
#
##vindex=[]
#vberita=[]
#vlabel=[]
#
#for row in rows:
#    vberita.append(row[1])
#    vlabel.append(row[2])
#
#X=vberita
#y=vlabel



df = pd.read_csv('E:/Digitalent Big Data/tugas pak ayok/cleandata-label.csv', encoding='utf-8')
df.head()

#a=df.label.value_counts()
#print(a)
a=df.dropna()

from sklearn.model_selection import train_test_split

X = a.berita
y = a.label



cv = ShuffleSplit(n_splits=10, test_size=0.2)


models = [
    LogisticRegression(),
    LinearSVC()
]

vect = TfidfVectorizer()
sm = SMOTE()

# Init a dictionary for storing results of each run for each model
results = {
    model.__class__.__name__: {
        'accuracy': [], 
        'f1_score': [],
        'confusion_matrix': []
    } for model in models
}

for train_index, test_index in cv.split(X):
    X_train, X_test  = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    X_train_vect = vect.fit_transform(X_train)    
    X_test_vect = vect.transform(X_test)
    
    X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)
    
    for model in models:
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test_vect)
        
        acc = accuracy_score(y_test, y_pred,average='micro')
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results[model.__class__.__name__]['accuracy'].append(acc)
        results[model.__class__.__name__]['f1_score'].append(f1)
        results[model.__class__.__name__]['confusion_matrix'].append(cm)
        
for model, d in results.items():
    avg_acc = sum(d['accuracy']) / len(d['accuracy'])
    avg_f1 = sum(d['f1_score']) / len(d['f1_score'])
    avg_cm = sum(d['confusion_matrix']) / len(d['confusion_matrix'])
    
    slashes = '-' * 30
    
    s = f"""{model}\n{slashes}
        Avg. Accuracy: {avg_acc:.2f}%
        Avg. F1 Score: {avg_f1:.3f}
        Avg. Confusion Matrix: 
        \n{avg_cm}
        """
    print(s)


#vectorizer = TfidfVectorizer(smooth_idf=False)
#tfidf = vectorizer.fit_transform(vberita)


#x_train=tfidf
#y_train=blabel
#
#clf = MultinomialNB()
#clf.fit(x_train, y_train)

#pickle.dump(vectorizer, open("C:/vectorizeridf.bin", "wb"))
#pickle.dump(clf, open("C:/clfidf.bin", "wb"))
#
#kf=KFold(n_splits=10, shuffle=True, random_state=0)
#print("k-fold cross val rata2: ",cross_val_score(clf, x_train, y_train, cv=kf, scoring='accuracy').mean()*100)
#print("k-fold cross val: ",cross_val_score(clf, x_train, y_train, cv=kf, scoring='accuracy')*100)
