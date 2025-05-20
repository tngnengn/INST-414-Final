import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("returns_cleaned.csv")

# ReturnTomorrow calculation
if 'ReturnTomorrow' not in df.columns:
    df['ReturnTomorrow'] = (df['NextClose'] - df['close']) / df['close']

# drop NaN
df = df.dropna(subset=['ReturnTomorrow'])


X = df[['DailyReturn', 'ThreeDayReturn']]
y = df['ReturnTomorrow']

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model training
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print the result
print("Linear Regression Results")
print("MSE:", mse)
print("MAE:", mae)
print("R² Score:", r2)

# save the result
linreg_results = pd.DataFrame([{
    "Model": "Linear Regression",
    "MSE": mse,
    "MAE": mae,
    "R2": r2
}])
linreg_results.to_csv("linreg_results.csv", index=False)
print("Saved the result")
