import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/CleanedNews.csv')

headline_counts = df.groupby('category')['headline'].count().sort_values(ascending=False)
print(headline_counts)

plt.figure(figsize=(12, 6))
sns.barplot(
    x=headline_counts.index,
    y=headline_counts.values,
    palette=sns.color_palette('tab10', n_colors=len(headline_counts))
)
plt.xlabel('Category')
plt.ylabel('Number of Headlines')
plt.title('Headline Count by Category')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

