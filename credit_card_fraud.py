# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 15:38:06 2022

@author: naqavi
"""

# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Loading the dataset
df = pd.read_csv('creditcard.csv')

# Separating features (X) and target variable (y)
X = df.iloc[:, 0:-1]  # Features
y = df.iloc[:, -1]    # Target variable (fraud or not fraud)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Standardizing the features using StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creating a Logistic Regression model
model = LogisticRegression(random_state=0)

# Fitting the model on the training data
clf = model.fit(X_train, y_train)

# Predicting probabilities for the test data
yhat = clf.predict_proba(X_test)
pos_prob = yhat[:, 1]

# Calculating the ratio of fraud cases in the dataset
fraud = len(y[y == 1]) / len(y)

# Calculating ROC AUC score
roc_auc = roc_auc_score(y_test, pos_prob)
print('Fraud ROC AUC %.3f' % roc_auc)

# Plotting precision-recall curve
pyplot.plot([0, 1], [fraud, fraud], linestyle='--', label='Fraud')  # Plotting the baseline for fraud
precision, recall, _ = precision_recall_curve(y_test, pos_prob)
pyplot.plot(recall, precision, marker='.', label='Logistic Regression')  # Plotting the precision-recall curve
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.legend()
pyplot.show()
