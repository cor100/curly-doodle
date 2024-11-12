import pandas as pd
import spacy
import string
import contractions
import json

nlp = spacy.load("en_core_web_sm")
file = 'MI Dataset.csv'
df = pd.read_csv(file)

print(df.columns)

# get contractiions
# obtained from https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/19794953#19794953
with open('contractions.json', 'r') as f:
    contractions_dict = json.load(f)

# Define function to expand contractions
def expand_contractions(text, contractions_dict=contractions_dict):
    for contraction, expansion in contractions_dict.items():
        text = text.replace(contraction, expansion)
    return text

def preprocess_text(text):
    # expand contractions
    text = expand_contractions(text)
    text = contractions.fix(text)

    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return lemmas 
df['processed_text'] = df['text'].apply(preprocess_text)
print(df[['author', 'text', 'processed_text']].head())


#save to csv file
df.to_csv('processed_dataset.csv', index=False)
