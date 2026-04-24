import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data/NewsCategorizer.csv')
    
# Display basic statistics
print(df.describe())

# Drop the keywords column
df = df.drop(columns=['keywords'])

# Check for duplicates
print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nDuplicate headlines:")
print(df['headline'].duplicated().sum())

# Remove rows with duplicate headlines
df = df.drop_duplicates(subset=['headline'])

# Display basic statistics
print(df.describe())