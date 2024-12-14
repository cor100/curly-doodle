from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset  
from sklearn.model_selection import train_test_split, MinMaxScaler
import pandas as pd
import numpy as np



# Reload your fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained("./finetuned_global_scores_model")
df = pd.read_csv('./csvs/proccesed_annomi.csv')
scores = pd.read_csv('./csvs/processed_global_scores.csv')

data = df.merge(scores, on="transcript_id", how="inner")
print(data.head())
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
target_columns = ["cultivating_change_talk", "softening_sustain_talk", "partnership", "empathy"]
train_data["labels"] = train_data[target_columns].values.tolist()
test_data["labels"] = test_data[target_columns].values.tolist()


#convert to use Hugging Face?
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print(train_dataset.column_names)

def tokenize_function(text):
    return tokenizer(text=text["utterance_text"], truncation=True, padding=True, max_length=512)
# data['tokenized'] = data['utterance_text']
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = ((predictions - labels) ** 2).mean(axis=0)  # Mean Squared Error
    mae = np.abs(predictions - labels).mean(axis=0)  # Mean Absolute Error
    baseline_mse = ((labels - labels.mean(axis=0)) ** 2).mean()

    return {
        "mse": mse.mean(),  # Average across all target columns
        "mae": mae.mean(),
        "Baseline MSE": baseline_mse
    }

# Define evaluation arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Evaluate
results = trainer.evaluate()
print(results)
