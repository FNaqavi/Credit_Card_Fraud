# Credit_Card_Fraud
Data is available in https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


# Credit Card Fraud Detection using Logistic Regression

This Python script demonstrates credit card fraud detection using Logistic Regression. It loads a dataset, preprocesses it, trains a Logistic Regression model, and evaluates its performance.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib

## Usage

1. Ensure you have the required libraries installed (`pandas`, `scikit-learn`, `matplotlib`).
2. Save your dataset as `creditcard.csv` in the same directory as the script.
3. Run the script in a Python environment.

## Description

- **Loading the Dataset**: The script loads the credit card transaction dataset from `creditcard.csv`.

- **Preprocessing**: It separates features (X) and the target variable (y) and then splits the dataset into training and testing sets. It also standardizes the features using `StandardScaler`.

- **Model Training**: It creates a Logistic Regression model and fits it on the training data.

- **Evaluation**: The model's performance is evaluated using the ROC AUC score. It also plots a precision-recall curve to visualize the precision-recall trade-off.

