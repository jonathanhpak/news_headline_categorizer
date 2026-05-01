import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/CleanedNews.csv')

df["word_length"] = df["headline"].str.split().str.len()
df["character_length"] = df["headline"].str.len()
print(df.head())

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1) 
sns.violinplot(data=df, x="category", y="word_length", palette="muted")
plt.title("Word Length Distribution by Category")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2) 
sns.violinplot(data=df, x="category", y="character_length", palette="magma")
plt.title("Character Length Distribution by Category")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()