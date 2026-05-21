import numpy as np
import pandas as pd
import re

#read dataset and drop unneeded columns
df = pd.read_csv('data/CleanedNews.csv')
df.drop(["links", "short_description"], axis=1, inplace=True)

print(df["category"].value_counts())