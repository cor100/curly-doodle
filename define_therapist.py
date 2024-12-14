import pandas as pd
import re

# Load the dataset
file_path = 'preprocessed_dataset.csv'  # Adjust as necessary
df = pd.read_csv(file_path)

# Group text by feature
feature_groups = df.groupby('final agreed label')['text'].apply(list)

