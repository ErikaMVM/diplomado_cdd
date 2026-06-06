import re
import spacy
import unicodedata
import nltk
from nltk.corpus import stopwords

# Descargar stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

nlp = spacy.load("es_core_news_sm", disable=["ner", "parser"])

def clean_text(text, pattern="[^a-záéíóúñüA-Z ]"):
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore')
    text = re.sub(pattern, "", text.decode("utf-8"), flags=re.UNICODE)
    text = u' '.join(text.lower().split())
    return text

def lemmatize_pipe(texts):
    docs = nlp.pipe(texts, disable=["ner", "parser"])
    return [" ".join([token.lemma_ for token in doc if not token.is_stop]) for doc in docs]

def get_stop_words():
    stop_words = stopwords.words("spanish")
    stop_words = [clean_text(word) for word in stop_words]
    return stop_words