import pandas as pd
import spacy
import string

nlp = spacy.load("en_core_web_sm")
file = 'MI Dataset.csv'
df = pd.read_csv(file)

print(df.columns)

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return lemmas 
df['processed_text'] = df['text'].apply(preprocess_text)
print(df[['author', 'text', 'processed_text']].head())


#save to csv file
df.to_csv('processed_dataset.csv', index=False)
