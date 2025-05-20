import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("returns_cleaned.csv")
plt.figure(figsize=(8, 4))
sns.histplot(df['DailyReturn'], bins=100, kde=True, color='skyblue')
plt.title("Distribution of Daily Return")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

labels = ['Down or Same (0)', 'Up (1)']
sizes = df['Target'].value_counts().sort_index()
colors = ['#ff9999','#66b3ff']

plt.figure(figsize=(5, 5))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Target Variable Distribution')
plt.axis('equal')
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
sns.boxplot(x='Target', y='ThreeDayReturn', data=df, palette='pastel')
plt.title("3-Day Return by Target Class")
plt.xlabel("Target (0 = Down or Same, 1 = Up)")
plt.ylabel("ThreeDayReturn")
plt.grid(True)
plt.tight_layout()
plt.show()
