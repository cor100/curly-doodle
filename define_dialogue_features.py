from textblob import TextBlob
from collections import Counter
from transformers import BertForSequenceClassification

import pandas as pd

df = pd.read_csv('processed_dataset.csv')

def sentiment_analysis(text):
    return TextBlob(text).sentiment.polarity

def extract_dialogue_features(dialogue):
    # from MITI 
    behaviors = {
        'Giving Information': 0,
        'Persuade': 0,
        'Persuade with Permission': 0,
        'Question': 0,
        'Simple Reflection': 0,
        'Complex Reflection': 0,
        'Affirm': 0,
        'Seeking Collaboration': 0,
        'Emphasizing Autonomy': 0,
        'Confront': 0
    }

    miti_mappings={
        "Cultivating": ["hope", "motivation", "enthusiasm"],
        "Softening":["anger","frustration","fear"], # want decrease
        "Partnership": ["trust","agreement"],
        "Empathy": ["warmth","understanding","caring"]
    }
    def classify(tokens):
        
    for behavior in behaviors():
        behaviors[behavior] += 1

df['sentiment'] = df['processed_text'].apply(sentiment_analysis)
df.to_csv('blob_dataset.csv', index=False)
