# INST-414-Final
# Predicting Stock Price Reversals Using Daily Market Indicators

This project explores whether short-term stock price reversals can be predicted using basic technical indicators such as daily and 3-day returns. The goal is to build classification and regression models that forecast the direction and magnitude of next-day price movement using historical S&P 500 data.

## Project File Structure

```
INST-414-Final/
├── data/                        # Original and cleaned datasets
│   ├── all_stocks_5yr.csv          # Raw S&P 500 historical data
│   └── returns_cleaned.csv         # Cleaned data with engineered features
│
├── codes_models/               # Modeling scripts and preprocessing
│   ├── logistic_regression.py      # Logistic Regression model
│   ├── randomforest.py             # Random Forest model
│   ├── modeltree.py                # Decision Tree model
│   ├── linearregression.py         # Linear Regression model
│   ├── compare_results.py          # Evaluation comparison script
│   └── EAD.py                      # Exploratory data analysis script
│
├── results/                    # Model outputs and visualizations
│   ├── log_results.csv             # Logistic regression metrics
│   ├── rf_results.csv              # Random forest metrics
│   ├── tree_results.csv            # Decision tree metrics
│   └── linreg_results.csv          # Linear regression metrics
│
└── README.md                  # Project overview and file documentation
```



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

