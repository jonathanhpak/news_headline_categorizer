import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

#IMPORT STRATIFIEDKFOLD, GRIDSEARCHCV, SEABORN, MATPLOTLIB
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

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



# Fine-Tuning with GridSearchCV
# Define possible hyperparameters to pass into the model and test in the GridSearchCV


# LIST POSSIBLE VALUES FOR HYPERPARAMETERS SPECIFIC TO YOUR MODEL
param_grid = {
    "n_estimators": [150, 200],
    "max_depth": [6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8]
}

# Define the type of cross-validation to use. StratifiedKFold preserves class proportions across folds.
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize GridSearchCV object with your model, grid of parameters, type of CV, and scoring metric
grid = GridSearchCV(
    model,
    param_grid,
    cv=kf,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=2 

)

# Fit to the training/validation data
grid.fit(X_train, y_train_enc)

# best_model is a trained model using the hyperparameter values that achieved the best cross-validation Macro F1 score
best_model = grid.best_estimator_
print(grid.best_params_)

# Evaluate on final model on test data
y_pred_enc = best_model.predict(X_test)
y_pred = le.inverse_transform(y_pred_enc) 
class_labels = le.classes_   


# Compute new performance metrics
print("\n\nBest Model Performance on Test Set:")
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nMacro F1:", f1_score(y_test, y_pred, average="macro"))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Graph 1: Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred, labels=class_labels)
cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100

plt.figure(figsize=(12, 8))
sns.heatmap(
    cm_percent,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels,
    cbar_kws={"label": "% of Actual Category"}
)
plt.title("Confusion Matrix - XGBoost")
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
    if category in class_labels
])

f1_df = f1_df.sort_values("f1_score", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=f1_df, x="category", y="f1_score")
plt.title("Per-Category F1 Scores - XGBoost")
plt.xlabel("Category")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
