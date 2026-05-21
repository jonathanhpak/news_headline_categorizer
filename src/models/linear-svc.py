import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

#IMPORT YOUR MODEL HERE
from sklearn.svm import LinearSVC


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
model = LinearSVC(
    C=1.0,
    penalty="l2",
    loss="squared_hinge",
    dual=False,
    tol=1e-4,
    max_iter=1000,
    class_weight="balanced",
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

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt


# LIST POSSIBLE VALUES FOR HYPERPARAMETERS SPECIFIC TO YOUR MODEL
param_grid = {
    "C": [0.1, 1, 5],
    "class_weight": [None, "balanced"]
}
# Define the type of cross-validation to use
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize GridSearchCV
grid = GridSearchCV(
    LinearSVC(dual=False, random_state=42, penalty="l2", loss="squared_hinge"),
    param_grid,
    cv=kf,
    scoring="f1_macro",
    n_jobs=-1
)

# Fit to the training data
grid.fit(X_train, y_train)

# best_model is a trained model using the best hyperparameters
best_model = grid.best_estimator_
print(grid.best_params_)

# Evaluate on test data
y_pred = best_model.predict(X_test)

# Compute performance metrics
print("\n\nBest Model Performance on Test Set:")
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nMacro F1:", f1_score(y_test, y_pred, average="macro"))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Graph 1: Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100

plt.figure(figsize=(12, 8))
sns.heatmap(
    cm_percent,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    xticklabels=best_model.classes_,
    yticklabels=best_model.classes_,
    cbar_kws={"label": "% of Actual Category"}
)
plt.title("Confusion Matrix - LinearSVC")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Graph 2: Per-category F1 score bar chart
report = classification_report(y_test, y_pred, output_dict=True)

f1_df = pd.DataFrame([
    {
        "category": category,
        "f1_score": scores["f1-score"]
    }
    for category, scores in report.items()
    if category in best_model.classes_
])

f1_df = f1_df.sort_values("f1_score", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=f1_df, x="category", y="f1_score")
plt.title("Per-Category F1 Scores - LinearSVC")
plt.xlabel("Category")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()