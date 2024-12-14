import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

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

scaler = MinMaxScaler()
score_columns = ["cultivating_change_talk", "softening_sustain_talk", "partnership", "empathy"]
session_scores[score_columns] = scaler.fit_transform(session_scores[score_columns])

# Find the highest global score per session and the corresponding metric
session_scores["highest_score_metric"] = session_scores[
    ["cultivating_change_talk", "softening_sustain_talk", "partnership", "empathy"]
].idxmax(axis=1)  # Finds the column with the highest score

session_scores["highest_score_value"] = session_scores[
    ["cultivating_change_talk", "softening_sustain_talk", "partnership", "empathy"]
].max(axis=1)  # Finds the value of the highest score

# Verify the results
print(session_scores[["dialog_id", "highest_score_metric", "highest_score_value"]].head())

# Extract the top three behaviors for each session
sorted_labels_per_session["top_three_behaviors"] = sorted_labels_per_session["sorted_agreed_labels"].apply(lambda x: x[:3])

# Verify the result
print(sorted_labels_per_session.head())

# Merge session scores with top behaviors
final_mapping = session_scores.merge(
    sorted_labels_per_session[["dialog_id", "top_three_behaviors"]],
    on="dialog_id",
    how="left"
)

# Verify the result
print(final_mapping[["dialog_id", "highest_score_metric", "highest_score_value", "top_three_behaviors"]].head())

behavior_analysis = final_mapping.explode("top_three_behaviors").groupby(
    ["highest_score_metric", "top_three_behaviors"]
).size().reset_index(name="count")

# Sort by count to find the most frequent behaviors for each metric
behavior_analysis = behavior_analysis.sort_values(by=["highest_score_metric", "count"], ascending=[True, False])

# Print results
print(behavior_analysis)

#graph
behavior_analysis = final_mapping.explode("top_three_behaviors").groupby(
    ["highest_score_metric", "top_three_behaviors"]
).size().reset_index(name="count")

sns.barplot(
    data=behavior_analysis,
    x="top_three_behaviors",
    y="count",
    hue="highest_score_metric"
)
plt.title("Top Behaviors by Highest Scoring Metric")
plt.xticks(rotation=45)
plt.show()
