import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from xgboost import XGBClassifier

# load dataset
df = pd.read_csv("data/HeadlinesWithFeatures.csv")

# define what we're predicting from and what we're predicting
X_engineered = df.drop(["headline", "category"], axis=1)  # number columns
X_text = df["headline"]                                    # headline text
y = df["category"]                                         # answer we want to predict

# split into 80% training and 20% testing
X_text_train, X_text_test, X_eng_train, X_eng_test, y_train, y_test = train_test_split(
    X_text,
    X_engineered,
    y,
    test_size=0.2,
    random_state=42,  # same split every run
    stratify=y        # each category equally represented in both splits
)

# convert headline text into numbers
# words that are unique to a headline score higher than common words
vectorizer = TfidfVectorizer(
    stop_words="english",  # ignore words like "the", "is", "and"
    lowercase=True,
    ngram_range=(1, 2),    # look at single words and two-word phrases
    min_df=2,              # ignore words that appear in only one headline
    max_features=5000      # only keep the 5000 most important words
)
X_text_train_tfidf = vectorizer.fit_transform(X_text_train)  # learn vocab from training data
X_text_test_tfidf = vectorizer.transform(X_text_test)        # apply same vocab to test data

# rescale all number columns to the same range so no single column dominates
scaler = StandardScaler()
X_eng_train_scaled = scaler.fit_transform(X_eng_train)  # learn scale from training data
X_eng_test_scaled = scaler.transform(X_eng_test)        # apply same scale to test data

# join text numbers and engineered numbers into one combined table
X_train = hstack([X_text_train_tfidf, csr_matrix(X_eng_train_scaled)])
X_test = hstack([X_text_test_tfidf, csr_matrix(X_eng_test_scaled)])

# convert category labels to numbers since XGBoost only understands numbers
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)  # learn mapping from training labels
y_test_enc = le.transform(y_test)        # apply same mapping to test labels

# set up model
model = XGBClassifier(
    n_estimators=150,       # number of trees to build
    max_depth=4,            # how deep each tree can grow
    learning_rate=0.1,      # how much each tree corrects the previous one
    subsample=0.8,          # use 80% of training data per tree to prevent overfitting
    eval_metric="mlogloss", # how the model measures its error during training
    tree_method="hist",     # faster training algorithm
    random_state=42         # same results every run
)

# train the model
model.fit(X_train, y_train_enc)

# predict categories for test headlines
y_pred_enc = model.predict(X_test)

# convert predicted numbers back to category names
y_pred = le.inverse_transform(y_pred_enc)

# evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nMacro F1:", f1_score(y_test, y_pred, average="macro"))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))