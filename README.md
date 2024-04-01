# Credit_Card_Fraud
Data is available in https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


This code is a Python script for credit card fraud detection using logistic regression. Here's a breakdown of what it's trying to achieve:

**Data Loading**: It imports the necessary libraries and loads a dataset named 'creditcard.csv'. This dataset contains features related to credit card transactions, including transaction amounts, timestamps, etc.

**Data Preprocessing**: The script splits the dataset into features (X) and the target variable (y), where 'X' contains all columns except the last one (which represents the target variable indicating fraud or not fraud), and 'y' contains the target variable.

**Data Splitting**: It further splits the data into training and testing sets using the train_test_split function from sklearn.model_selection, with a test size of 10% of the total data and a random state for reproducibility.

**Feature Standardization**: The features are standardized using StandardScaler from sklearn.preprocessing. Standardization helps in bringing all features to a similar scale, which is a common practice in many machine learning algorithms.

**Model Building and Training**: It creates a logistic regression model using LogisticRegression from sklearn.linear_model, and fits this model to the training data.

**Model Evaluation**: The model predicts probabilities of fraud for the test data using predict_proba and extracts the probabilities of the positive class (fraud). It then calculates the ROC AUC score using roc_auc_score from sklearn.metrics to evaluate the model's performance.

**Visualization**: It plots the precision-recall curve using precision_recall_curve from sklearn.metrics and matplotlib.pyplot. This curve shows the trade-off between precision and recall for different thresholds. Additionally, it plots a dashed line representing the baseline for fraud, which is the ratio of actual fraud cases in the dataset.
