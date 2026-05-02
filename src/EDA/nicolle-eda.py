import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/CleanedNews.csv')

# New Columns for digit counts
import re
def has_n_digit_num(headline, n):
    if not isinstance(headline, str):
        return 0
    pattern = r'(?<!\d)\d{' + str(n) + r'}(?!\d)'
    return 1 if re.search(pattern, headline) else 0
 
df['has_1_digit_num'] = df['headline'].apply(lambda x: has_n_digit_num(x, 1)).astype(int)
df['has_2_digit_num'] = df['headline'].apply(lambda x: has_n_digit_num(x, 2)).astype(int)
df['has_3_digit_num'] = df['headline'].apply(lambda x: has_n_digit_num(x, 3)).astype(int)
df['has_4_digit_num'] = df['headline'].apply(lambda x: has_n_digit_num(x, 4)).astype(int)
 
print("\nColumn totals:")
print(df[['has_1_digit_num','has_2_digit_num','has_3_digit_num','has_4_digit_num']].sum())

#Graph
import numpy as np

grouped = df.groupby('category')[
    ['has_1_digit_num','has_2_digit_num','has_3_digit_num','has_4_digit_num']
].mean().reset_index()

x = np.arange(len(grouped['category']))
width = 0.2

cols = ['has_1_digit_num','has_2_digit_num','has_3_digit_num','has_4_digit_num']
labels = ['1-digit','2-digit','3-digit','4-digit']
colors = ['blue', 'green', 'orange', 'red']

plt.figure(figsize=(14,6))
for i in range(4):
    plt.bar(x + i*width, grouped[cols[i]] * 100, width, label=labels[i], color=colors[i])

plt.xticks(x + 1.5*width, grouped['category'], rotation=35, ha='right')
plt.ylabel('Percentage of headlines')
plt.title('Percentage of headlines by digit length and category')
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()