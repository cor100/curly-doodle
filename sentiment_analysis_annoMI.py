import pandas as pd
import pandas as pd
import numpy as np
import os
from datasets import Dataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, DataCollatorWithPadding, AutoTokenizer, TrainingArguments, Trainer
from huggingface_hub import login

LABELS = 4

df = pd.read_csv('./csvs/proccesed_annomi.csv')
scores = pd.read_csv('./csvs/processed_global_scores.csv')
texts = df.groupby('transcript_id').agg({
    "utterance_text": lambda x: ' '.join(x)
}).reset_index()
print(texts.head())

data = texts.merge(scores, on="transcript_id", how="inner")
print(data.head())
# from huggingface_hub import notebook_login
# notebook_login()
# login(token=os.getenv("HUGGINGFACETTOKEN"))
#### prepare dataset

# from the hugging face website https://huggingface.co/blog/sentiment-analysis-python
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
target_columns = ["cultivating_change_talk","partnership","softening_sustain_talk","empathy"]
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

### define model and tune
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = LABELS)

# Metric computation function
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



training_args = TrainingArguments(
   output_dir="./results",
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=2,
   weight_decay=0.01,
   save_strategy="epoch",
)

trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_train,
   eval_dataset=tokenized_test,
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("./finetuned_global_scores_model")

results = trainer.evaluate()
print(results)