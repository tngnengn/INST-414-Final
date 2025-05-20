import pandas as pd


df = pd.read_csv("all_stocks_5yr.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['Name', 'date'])

# DailyReturn calculation
df['DailyReturn'] = df.groupby('Name')['close'].pct_change()

# ThreeDayReturn calculation
df['ThreeDayReturn'] = df.groupby('Name')['DailyReturn'].rolling(window=3).sum().reset_index(level=0, drop=True)

# target
df['NextClose'] = df.groupby('Name')['close'].shift(-1)
df['Target'] = (df['NextClose'] > df['close']).astype(int)

# cleaning
df_clean = df.dropna(subset=['DailyReturn', 'ThreeDayReturn', 'Target'])

# save
df_clean.to_csv("returns_cleaned.csv", index=False)
print("download returns_cleaned.csv")