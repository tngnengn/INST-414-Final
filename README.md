# INST-414-Final
# Predicting Stock Price Reversals Using Daily Market Indicators

This project explores whether short-term stock price reversals can be predicted using basic technical indicators such as daily and 3-day returns. The goal is to build classification and regression models that forecast the direction and magnitude of next-day price movement using historical S&P 500 data.

## file

 INST-414-Final/
├──  data/                     # Original and cleaned datasets
│   ├── all_stocks_5yr.csv
│   └── returns_cleaned.csv
│
├──  codes_models/             # Modeling scripts and preprocessing
│   ├── logistic_regression.py
│   ├── randomforest.py
│   ├── modeltree.py
│   ├── linearregression.py
│   ├── compare_results.py
│   └── EAD.py
│
├──  results/                  # Model outputs and evaluation metrics
│   ├── log_results.csv
│   ├── rf_results.csv
│   ├── tree_results.csv
│   └── linreg_results.csv
│
├── README.md                   # Project overview and structure



## Models
- Logistic Regression  
- Random Forest  
- Decision Tree  
- Linear Regression

## Evaluation Metrics
- Accuracy, Precision, Recall, AUC (classification)  
- MSE, MAE, R² (regression)

## Dataset Source
Kaggle S&P 500 Stock Data (2013–2018):  
https://www.kaggle.com/datasets/camnugent/sandp500

