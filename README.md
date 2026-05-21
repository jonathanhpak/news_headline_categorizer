# News Headline Categorizer

A collaborative machine learning project focused on classifying news headlines into topic categories using natural language processing (NLP), engineered headline features, and multiple supervised learning models.

This project was developed through UCLA's National Student Data Corps (NSDC).

## Project Overview

The goal of this project was to build and evaluate models that can automatically classify short news headlines into one of several categories. Because headlines are brief and information-dense, this project focused on identifying which text-based and structural features are most useful for distinguishing between different types of news content.

The project explores questions such as:

- How well can machine learning models classify news headlines using only headline text?
- Which categories are easiest or hardest to distinguish?
- Which model types perform best on sparse, high-dimensional text data?
- Do engineered headline features improve performance beyond TF-IDF alone?

## Categories

The dataset includes headlines from 10 news categories:

- BUSINESS
- ENTERTAINMENT
- FOOD & DRINK
- PARENTING
- POLITICS
- SPORTS
- STYLE & BEAUTY
- TRAVEL
- WELLNESS
- WORLD NEWS

## Dataset

The project uses a HuffPost news headline dataset containing roughly 50,000 headlines and associated category labels.

Key dataset files:

```text
data/
├── NewsCategorizer.csv          # Original dataset
├── CleanedNews.csv              # Cleaned dataset
└── HeadlinesWithFeatures.csv    # Dataset with engineered features
```

## Exploratory Data Analysis

EDA focused on understanding category distributions and identifying linguistic patterns across categories. Analyses included:

- Category count distribution
- Word length and character length distributions
- Common words by category
- Sentiment distribution by category
- Punctuation, digit, and capitalization patterns

These analyses helped guide feature engineering and provided context for model performance.

## Feature Engineering

### Text Features

Headline text was transformed using TF-IDF vectorization, which assigns higher weights to words or phrases that are frequent in a headline but relatively rare across the full dataset.

TF-IDF preprocessing included:

- Lowercasing
- English stopword removal
- Unigrams and bigrams
- Minimum document frequency filtering

### Engineered Features

Additional headline-level features included:

- Word length
- Character length
- Presence of punctuation symbols such as `$`, `%`, `?`, `!`, `:`, `#`, and `-`
- Presence of 1-digit, 2-digit, 3-digit, and 4-digit numbers
- Presence of all-caps words
- Sentiment analysis features using TextBlob

Engineered numerical features were scaled before being combined with the sparse TF-IDF matrix.

## Models Evaluated

The project compared several supervised machine learning models:

| Model Group | Models |
|---|---|
| Linear Models | Logistic Regression, LinearSVC |
| Tree-Based Models | Random Forest, XGBoost |
| Neural Network | Multi-Layer Perceptron (MLP) |
| Distance-Based Model | K-Nearest Neighbors (KNN) |

Each model used the same train/test split and evaluation metrics to keep comparisons consistent.

## Model Tuning and Evaluation

Models were evaluated using:

- Accuracy
- Macro F1 score
- Classification reports
- Confusion matrix heatmaps
- Per-category F1 score bar charts

Hyperparameter tuning was performed using GridSearchCV with stratified K-fold cross-validation.

## Final Results

| Model | Final Accuracy | Final Macro F1 |
|---|---:|---:|
| Logistic Regression | 0.758 | 0.755 |
| LinearSVC | 0.768 | 0.764 |
| MLP | 0.770 | 0.765 |
| XGBoost | 0.663 | 0.664 |
| Random Forest | 0.613 | 0.606 |
| KNN | 0.467 | 0.459 |

Among the evaluated models, LinearSVC and MLP achieved the strongest overall performance on the classification task.

The project demonstrated that relatively simple linear models can perform extremely well on sparse NLP classification tasks when paired with strong text representations such as TF-IDF.

## Repository Structure

```text
news_headline_categorizer/
├── data/                  # Raw, cleaned, and feature-engineered datasets
├── src/
│   ├── EDA/
│   │   ├── build-features.py
│   │   └── individual EDA scripts
│   └── models/
│       ├── logistic-regression.py
│       ├── linear-svc.py
│       ├── random-forest.py
│       ├── xgboost.py
│       ├── mlp.py
│       └── knn.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- SciPy
- matplotlib
- seaborn
- TextBlob

## How to Run

Clone the repository and install the required Python packages.

```bash
git clone https://github.com/jonathanhpak/news_headline_categorizer
cd news_headline_categorizer
pip install -r requirements.txt
```

Run an individual model script from the project root. For example:

```bash
python src/models/logistic-regression.py
```

## Contributors

This project was completed collaboratively by members of UCLA NSDC.