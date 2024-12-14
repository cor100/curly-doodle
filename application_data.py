import pandas as pd
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
df = pd.read_csv('./csvs/processed_dataset.csv')
scores=pd.read_csv('./csvs/global_scores.csv')

tokenizer = AutoTokenizer.from_pretrained("./finetuned_global_scores_model")
model = AutoModelForSequenceClassification.from_pretrained("./finetuned_global_scores_model")
model.eval()  # Set the model to evaluation mode

label_counts = df.groupby(["dialog_id", "final agreed label"]).size().reset_index(name="count")
labels = ['Give Information','Open Question','Closed Question', 'Simple Reflection', 'Closed Reflection', 'Affirm','Emphasize Autonomy','Confront']
# Ensure all possible labels are represented in each session
#  cartesian product of dialog_id and possible labels
unique_sessions = df["dialog_id"].unique()
session_label_combinations = pd.MultiIndex.from_product(
    [unique_sessions, labels], names=["dialog_id", "final agreed label"]
).to_frame(index=False)
label_counts_complete = session_label_combinations.merge(
    label_counts, on=["dialog_id", "final agreed label"], how="left"
)

# Fill missing counts with 0 for labels not represented in a session
label_counts_complete["count"] = label_counts_complete["count"].fillna(0).astype(int)

# Verify the result
print(label_counts_complete.head())



# sort labels by count
sorted_labels = label_counts_complete.sort_values(by=["dialog_id", "count"], ascending=[True, False])

# aggregate features
sorted_labels_per_session = sorted_labels.groupby("dialog_id")["final agreed label"].apply(list).reset_index()
sorted_labels_per_session.columns = ["dialog_id", "sorted_agreed_labels"]


print(sorted_labels_per_session.head())
session_scores = scores.merge(sorted_labels_per_session, on='dialog_id', how='left')
print(session_scores.head())
session_scores.to_csv('./csvs/session_scores.csv', index=False)
