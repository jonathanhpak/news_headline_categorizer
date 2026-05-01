import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

#import logistic regression and metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#read dataset and drop unneeded columns
df = pd.read_csv("data/FeaturedNews.csv")
X_engineered = df.drop(["headline", "category"], axis=1)


scaler = StandardScaler()
X_engineered_scaled = scaler.fit_transform(X_engineered)