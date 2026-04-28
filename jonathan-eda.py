import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

df = pd.read_csv('data/CleanedNews.csv')
print(df.shape)

#create sentiment column
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df["sentiment"] = df["headline"].apply(get_sentiment)

#visualize sentiment distribution
sentiment_counts = df["sentiment"].value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'})
plt.title('Sentiment Distribution of News Headlines')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

#visualize sentiment distribution by category
sentiment_pct = (
    df.groupby("category")["sentiment"]
    .value_counts(normalize=True)
    .unstack()
)

sentiment_pct.plot(kind="bar", stacked=True)
plt.title("Sentiment Percentage per Category")
plt.xticks()
plt.tight_layout()
plt.show()