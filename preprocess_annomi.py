import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# from https://huggingface.co/tasks/sentence-similarity
from sentence_transformers import SentenceTransformer, util

df = pd.read_csv("./csvs/proccesed_annomi.csv")
LABEL = 4

df["cultivating_change_talk"] = 0
df["softening_sustain_talk"] = 0
df["partnership"] = 0
df["empathy"] = 0


    
df["cultivating_change_talk"] = df["client_talk_type"].apply(lambda x: 1 if x == "change" else 0)
df["softening_sustain_talk"] = df["client_talk_type"].apply(lambda x: 1 if x != "sustain" else 0)
df["partnership"] = df.apply(
    lambda row: 1 if row["question_subtype"] == "open" or row["therapist_input_subtype"] in ["information", "negotiation", "options"] or row["client_talk_type"] == "neutral" else 0,
    axis=1,
)

# EMPATHY MAPPING
# sentiment analysis
analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

df["client_sentiment"] = 0.0
df["therapist_sentiment"] = 0.0
df["sentiment_mismatch"] = 0.0
df["semantic_similarity"] = 0.0
df["is_reflection"] = 0
df["is_closed"] = 0
df["misinterpretation_score"] = 0.0

validation_phrases = ['I understand', 'That makes sense', 'I appreciate', 'good suggestion', 'great']
for i in range(0, df.shape[0] - 1, 2):
    client_row = df.iloc[i]
    therapist_row = df.iloc[i+1]

    if client_row["interlocutor"] == "client" and therapist_row["interlocutor"] == "therapist":
        client_utterance = client_row["utterance_text"]
        therapist_response = therapist_row["utterance_text"]
        
        #reflective listening
        if therapist_row['main_therapist_behaviour'] == 'reflection':
            df.at[i+1, 'is_reflection'] = 1
        else:
            df.at[i+1, 'is_reflection'] = 0
        if therapist_row['question_subtype'] == 'closed':
            df.at[i+1, 'is_closed'] = 1
        else:
            df.at[i+1, 'is_closed'] = 0

        #sentiment analysis
        try:
            client_sentiment = analyzer(client_utterance)[0]["label"]
            therapist_sentiment = analyzer(therapist_response)[0]["label"]
            client_score = 1 if client_sentiment == "POSITIVE" else -1
            therapist_score = 1 if therapist_sentiment == "POSITIVE" else -1
            df.at[i, "client_sentiment"] = client_score
            df.at[i + 1, "therapist_sentiment"] = therapist_score

            # difference between sentiment
            df.at[i + 1, "sentiment_mismatch"] = abs(client_score - therapist_score)
        except Exception as e:
            print(f"Error processing sentiment for rows {i}-{i+1}: {e}")
        

        try:
            client_embed = similarity_model.encode(client_utterance, convert_to_tensor=True)
            therapist_embed = similarity_model.encode(therapist_response, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(client_embed, therapist_embed).item()
            df.at[i+1, 'semantic_similarity'] = similarity
        except Exception as e:
            print(f"Error processing similarity for rows {i}-{i+1}: {e}")

        #validation and affirmation
        if any(phrase in therapist_response.lower() for phrase in validation_phrases):
            df.at[i + 1, "is_validation"] = 1


metrics = df.groupby('transcript_id').agg({
    'is_reflection':'sum',
    'utterance_text':'count',
    'is_validation': 'sum',
    'is_closed': sum
}).reset_index()

metrics['reflection_ratio'] = metrics['is_reflection'] / metrics['utterance_text']
metrics['validation_ratio'] = metrics['is_validation'] / metrics['utterance_text']
metrics['closed_ratio'] = metrics['is_closed'] / metrics['utterance_text']
df = df.merge(metrics[['transcript_id', 'reflection_ratio', 'validation_ratio', 'closed_ratio']], on='transcript_id')

df['empathy'] = (
    ((1-df['sentiment_mismatch']) + df['semantic_similarity'] +
      (1-df['misinterpretation_score']) + df['reflection_ratio'] +
        (1-metrics["closed_ratio"]) + df['validation_ratio']) / 6.0 
)
 
aggregates = df.groupby('transcript_id')[['cultivating_change_talk', 'partnership','softening_sustain_talk','empathy']].mean().reset_index()
aggregates.to_csv('./csvs/processed_global_scores.csv', index=False)

print(aggregates.head())
 