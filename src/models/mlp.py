import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

#IMPORT YOUR MODEL HERE
from sklearn.neural_network import MLPClassifier


#read dataset and drop unneeded columns
df = pd.read_csv("data/HeadlinesWithFeatures.csv")

#define features and target
X_engineered = df.drop(["headline", "category"], axis=1)
X_text = df["headline"]
y = df["category"]

#split data into train and test sets (keep random_state = 42 to ensure you get the same data splits every time.)
X_text_train, X_text_test, X_eng_train, X_eng_test, y_train, y_test = train_test_split(
    X_text,
    X_engineered,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#tf-idf vectorization
# converts headline text into numerical features by assigning higher weights
# to words that are frequent in a headline but rare across all headlines.
# this captures the most informative words and serves as the main signal
# for the model to distinguish between categories!
vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)
X_text_train_tfidf = vectorizer.fit_transform(X_text_train)
X_text_test_tfidf = vectorizer.transform(X_text_test)

#scale engineered features (i.e. change them to be on the same scale)
scaler = StandardScaler()
X_eng_train_scaled = scaler.fit_transform(X_eng_train)
X_eng_test_scaled = scaler.transform(X_eng_test)

#combine tf-idf features with engineered features
X_train = hstack([X_text_train_tfidf, csr_matrix(X_eng_train_scaled)])
X_test = hstack([X_text_test_tfidf, csr_matrix(X_eng_test_scaled)])

#instantiate model
#REPLACE WITH YOUR MODEL AND BASELINE HYPERPARAMETERS! Keep random_state = 42 for reproducibility.
model = MLPClassifier(
    hidden_layer_sizes=(64, 32), # Two layers: 64 neurons, then 32
    activation='relu',           # Standard non-linearity
    solver='adam',               # Best general-purpose solver
    alpha=0.001,                 # Light regularization
    learning_rate_init=0.001,    # Default learning rate
    max_iter=500,                # Give it more time to converge
    random_state=42
)

#train model on training data
model.fit(X_train, y_train)

#predict
y_pred = model.predict(X_test)

#evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nMacro F1:", f1_score(y_test, y_pred, average="macro"))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))