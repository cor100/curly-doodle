import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.phrases import Phrases, Phraser
import string
import pandas as pd
import json

nltk.download('punkt_tab')
nltk.download('wordnet')
file = './csvs/MI Dataset.csv'
file2 = './csvs/AnnoMI-full.csv'

df = pd.read_csv(file)
df_anno = pd.read_csv(file2)

# print(df.columns)
lemmatizer = WordNetLemmatizer()
import contractions

# get contractiions
# obtained from https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/19794953#19794953
with open('contractions.json', 'r') as f:
    contractions_dict = json.load(f)

# only high quality
df_anno = df_anno[df_anno['mi_quality'] == 'high']

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
    tokens = simple_preprocess(text, deacc=True) 
    tokens =[lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if token not in STOPWORDS and token.isalpha()]

    bigram_model = Phrases([tokens], min_count = 1, threshold = 2)
    bigram_phraser = Phraser(bigram_model)
    tokens = bigram_phraser[tokens]

    tokens = [token for token in tokens if len(token) > 2]
    return tokens

def preprocess_text2(text):
    text = expand_contractions(text)
    text = contractions.fix(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in STOPWORDS and token.isalpha()]

    
    # Optional: Join tokens back into a string (for models requiring raw text)
    processed_text = ' '.join(tokens)
    return processed_text

df['processed_text'] = df['text'].apply(preprocess_text)
df_anno['processed_text'] = df_anno['utterance_text'].apply(preprocess_text2)

print(df[['author', 'text', 'processed_text']].head())
print(df_anno[['video_title', 'processed_text']].head())


#save to csv file
df.to_csv('processed_dataset.csv', index=False)
df_anno.to_csv('proccesed_annomi.csv', index=False)
