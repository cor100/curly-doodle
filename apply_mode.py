import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
df = pd.read_csv('./csvs/processed_dataset.csv')
tokenizer = AutoTokenizer.from_pretrained("./finetuned_global_scores_model")
model = AutoModelForSequenceClassification.from_pretrained("./finetuned_global_scores_model")
model.eval()  # Set the model to evaluation mode

def preprocess_for_inference(row):
    return tokenizer(
        row["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=512, 
        return_tensors="pt"
    )

texts = df.groupby('dialog_id').agg({
    "text": lambda x: ' '.join(x)
}).reset_index()
texts['tokenized_text'] = texts.apply(preprocess_for_inference, axis=1)
print(texts.head())

global_scores = []

# Iterate over tokenized data and predict
for idx, row in texts.iterrows():
    inputs = row["tokenized_text"]
    with torch.no_grad():  # Disable gradient computation
        outputs = model(**inputs)
        scores = outputs.logits.squeeze().tolist()  # Get global scores
        global_scores.append(scores)

texts[["cultivating_change_talk", "softening_sustain_talk", "partnership", "empathy"]] = pd.DataFrame(global_scores)
print(texts.head())
texts.to_csv('./csvs/global_scores.csv', index=False)