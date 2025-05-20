import pandas as pd
import matplotlib.pyplot as plt

rf = pd.read_csv("rf_results.csv")
log = pd.read_csv("log_results.csv")
tree = pd.read_csv("tree_results.csv")

rf['Model'] = 'Random Forest'
log['Model'] = 'Logistic Regression'
tree['Model'] = 'Decision Tree'

df = pd.concat([rf, log, tree], ignore_index=True)
df = df[['Model', 'Accuracy', 'Precision', 'Recall', 'AUC']]

metrics = ['Accuracy', 'Precision', 'Recall', 'AUC']
model_names = df['Model']
plt.figure(figsize=(10, 6))
bar_width = 0.2
x = range(len(model_names))

for i, metric in enumerate(metrics):
    plt.bar([p + bar_width * i for p in x], df[metric], width=bar_width, label=metric)

plt.xlabel("Model")
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.xticks([p + bar_width * 1.5 for p in x], model_names)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()  