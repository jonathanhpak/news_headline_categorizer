import numpy as np
import pandas as pd
import re

#read dataset and drop unneeded columns
df = pd.read_csv('data/CleanedNews.csv')
df.drop(["links", "short_description"], axis=1, inplace=True)

#length features
df["word_length"] = df["headline"].str.split().str.len()
df["character_length"] = df["headline"].str.len()

#character features
df['has_dollar_sign'] = df['headline'].str.contains("$", regex = False).astype(int)
df['has_percent'] = df['headline'].str.contains("%", regex = False).astype(int)
df['has_question_mark'] = df['headline'].str.contains("?", regex = False).astype(int)
df['has_exclamation_mark'] = df['headline'].str.contains("!", regex = False).astype(int)
df['has_colon'] = df['headline'].str.contains(":", regex = False).astype(int)
df['has_hashtag'] = df['headline'].str.contains("#", regex = False).astype(int)
df['has_dash'] = df['headline'].str.contains("-", regex = False).astype(int)

#digit features
def has_n_digit_num(headline, n):
    if not isinstance(headline, str):
        return 0
    pattern = r'(?<!\d)\d{' + str(n) + r'}(?!\d)'
    return 1 if re.search(pattern, headline) else 0
 
df['has_1_digit_num'] = df['headline'].apply(lambda x: has_n_digit_num(x, 1)).astype(int)
df['has_2_digit_num'] = df['headline'].apply(lambda x: has_n_digit_num(x, 2)).astype(int)
df['has_3_digit_num'] = df['headline'].apply(lambda x: has_n_digit_num(x, 3)).astype(int)
df['has_4_digit_num'] = df['headline'].apply(lambda x: has_n_digit_num(x, 4)).astype(int)

#capitalization features
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
df['has_all_caps_word'] = has_all_caps_word(df)

#create csv with features
df.to_csv("data/HeadlinesWithFeatures.csv", index=False)