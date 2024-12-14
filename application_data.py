import pandas as pd
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
df = pd.read_csv('./csvs/processed_dataset.csv')
tokenizer = AutoTokenizer.from_pretrained("./finetuned_global_scores_model")
model = AutoModelForSequenceClassification.from_pretrained("./finetuned_global_scores_model")
model.eval()  # Set the model to evaluation mode

label_counts = df.groupby(["dialog_id", "final agreed label"]).size().reset_index(name="count")
labels = ['Give Information','Open Question','Closed Question', 'Simple Reflection', 'Closed Reflection', 'Affirm','Emphasize Autonomy','Confront']
# Ensure all possible labels are represented in each session
# Create a full cartesian product of dialog_id and possible labels
unique_sessions = df["dialog_id"].unique()
session_label_combinations = pd.MultiIndex.from_product(
    [unique_sessions, labels], names=["dialog_id", "final agreed label"]
).to_frame(index=False)

# Merge actual counts with the full combinations
label_counts_complete = session_label_combinations.merge(
    label_counts, on=["dialog_id", "final agreed label"], how="left"
)

# Fill missing counts with 0 for labels not represented in a session
label_counts_complete["count"] = label_counts_complete["count"].fillna(0).astype(int)

# Verify the result
print(label_counts_complete.head())



# Sort labels by count within each session
sorted_labels = label_counts_complete.sort_values(by=["dialog_id", "count"], ascending=[True, False])

# Aggregate the sorted labels into a list for each session
sorted_labels_per_session = sorted_labels.groupby("dialog_id")["final agreed label"].apply(list).reset_index()

# Rename columns for clarity
sorted_labels_per_session.columns = ["dialog_id", "sorted_agreed_labels"]

# Verify the result
print(sorted_labels_per_session.head())