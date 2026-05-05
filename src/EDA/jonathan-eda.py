import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

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
plt.title('Sentiment Label Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

#visualize sentiment distribution by category
sentiment_pct = (
    df.groupby("category")["sentiment"]
    .value_counts(normalize=True)
    .unstack()
)

sentiment_pct.plot(kind="bar", stacked=True, color={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'})
plt.title("Sentiment Label Distribution by Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()





#tf-idf
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2), min_df=2)
tfidf_matrix = vectorizer.fit_transform(df['headline'])
feature_names = vectorizer.get_feature_names_out()

#print top words for each category
def top_tfidf_words(category, n=10):
    row_indices = np.where(df["category"].values == category)[0]
    mean_scores = tfidf_matrix[row_indices].mean(axis=0).A1
    
    top_indices = np.argsort(mean_scores)[::-1][:n]
    
    words = [feature_names[i] for i in top_indices]
    return words

for cat in df["category"].unique():
    print(f"\nTop words for {cat}:")
    print(", ".join(top_tfidf_words(cat)))

#plotting
def plot_top_tfidf(category, n=10):
    row_indices = np.where(df["category"].values == category)[0]
    mean_scores = tfidf_matrix[row_indices].mean(axis=0).A1

    top_indices = np.argsort(mean_scores)[::-1][:n]

    words = [feature_names[i] for i in top_indices]
    scores = [mean_scores[i] for i in top_indices]

    plot_df = pd.DataFrame({
        "word": words,
        "score": scores
    })

    plot_df = plot_df.sort_values("score")  # for horizontal bar

    plt.figure(figsize=(8,6))
    plt.barh(plot_df["word"], plot_df["score"])
    plt.title(f"Top TF-IDF Words: {category}")
    plt.xlabel("TF-IDF Score")
    plt.tight_layout()
    plt.show()

# top words for sports/wellness (visualization)
plot_top_tfidf("SPORTS")
plot_top_tfidf("WELLNESS")