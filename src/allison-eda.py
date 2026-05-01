import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/CleanedNews.csv')
print(df.head())

df['has_dollar_sign'] = df['headline'].str.contains("$", regex = False).astype(int)
df['has_percent'] = df['headline'].str.contains("%", regex = False).astype(int)
df['has_question_mark'] = df['headline'].str.contains("?", regex = False).astype(int)
df['has_exclamation_mark'] = df['headline'].str.contains("!", regex = False).astype(int)
df['has_colon'] = df['headline'].str.contains(":", regex = False).astype(int)
df['has_hashtag'] = df['headline'].str.contains("#", regex = False).astype(int)
df['has_dash'] = df['headline'].str.contains("-", regex = False).astype(int)

char_columns = [
    'has_dollar_sign', 'has_percent', 'has_question_mark', 
    'has_exclamation_mark', 'has_colon', 'has_hashtag', 'has_dash'
]

heatmap_data = df.groupby("category")[char_columns].mean() * 100
heatmap_data.columns = ['$', '%', '?', '!', ':', '#', '-']
sns.heatmap(heatmap_data, annot=True)
plt.title('Percentage of Headlines Containing Specific Characters by Category')
plt.show()