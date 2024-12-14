import pandas as pd
import numpy as np
import os
import evaluate
from datasets import Dataset
from sklearn.utils import shuffle
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from huggingface_hub import login

# from huggingface_hub import notebook_login
# notebook_login()
login(token=os.getenv("HUGGINGFACETTOKEN"))

data_collator = DataCollatorWithPadding()
df = pd.read_csv('processed_dataset_annoMI.csv')

# from the hugging face website https://huggingface.co/blog/sentiment-analysis-python
df_shuffled = shuffle(df, random_state=42)
TRAIN_SIZE = 3000
TEST_SIZE = 300

#convert to use Hugging Face?
small_train_dataset = df_shuffled.iloc[:TRAIN_SIZE]
small_test_dataset = df_shuffled.iloc[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]
train_dataset = Dataset.from_pandas(small_train_dataset)
test_dataset = Dataset.from_pandas(small_test_dataset)

# tokenizer to use for Hugging Face models
tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
# num_labels: specifies the number of distinct labels that the model predicts
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

# speed up training: use a data_collator to convert training samples to PyTorch tensors
# used in Trainer
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    load_accuracy = evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}

def tokenize_function(text):
    return tokenizer(text["processed_text"], truncation=True)
    
target_columns = ["cultivating_change_talk", "softening_sustain_talk", "partnership", "empathy"]

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

repo_name = "finetuning-sentiment-model-3000-samples"
training_args = TrainingArguments(
   output_dir=repo_name,
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=2,
   weight_decay=0.01,
   save_strategy="epoch",
   push_to_hub=True,
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
trainer.evaluate()