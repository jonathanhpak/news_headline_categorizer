import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def has_all_caps_word(df):
    s = df['headline']
    r = []
    for x in range(45577):
        headline = s[x].split()
        found = False
        for word in headline:
            if len(word) > 1 and word.isupper():
                r.append(1)
                found = True
                break
        if not found:
            r.append(0)
    return r

def find_unique_categories(df):
    s = df['category']
    unique_categories = []
    for x in range(45577):
        if not s[x] in unique_categories:
            unique_categories.append(s[x])
    return unique_categories

def count_matches(df):
    cats = df['category']
    matches = df['has_all_caps_word']
    uniques = find_unique_categories(df)
    totals = []
    for u in uniques:
        total = 0
        for x in range (45577):
            if cats[x] == u:
                total  += matches[x]
        totals.append(total)
    return totals

df = pd.read_csv('data/CleanedNews.csv')
df['has_all_caps_word'] = has_all_caps_word(df)

percentages = (
    df.groupby("category")["has_all_caps_word"]
    .mean() * 100
)

x = percentages.index
y = percentages.values

plt.figure(figsize=(10, 6))
plt.bar(x, y)
plt.title("Percentage of Headlines with All-Caps Words by Category")
plt.xlabel("Category")
plt.ylabel("Percentage of Headlines")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
