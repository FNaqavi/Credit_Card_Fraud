# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 15:38:06 2022

@author: naqavi
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

from sklearn.linear_model import LogisticRegression


df = pd.read_csv('creditcard.csv')

X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.1, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

model = LogisticRegression(random_state=0)

clf = model.fit(X_train, y_train)
yhat = clf.predict_proba(X_test)
pos_prob = yhat[:,1]
fraud = len(y[y==1]) / len(y)
roc_auc = roc_auc_score(y_test, pos_prob)
print('Fraud ROC AUC %.3f' % roc_auc)
pyplot.plot([0, 1], [fraud, fraud], linestyle='--', label='Fraud')
precision, recall, _ = precision_recall_curve(y_test, pos_prob)
pyplot.plot(recall, precision, marker='.', label='Decision_Tree')
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.legend()
pyplot.show()
