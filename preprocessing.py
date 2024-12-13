import re
import nltk
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+', '', text.lower())  # Clean URLs & Lowercase
    text = re.sub(r'\W', ' ', text)  # Remove punctuation
    tokens = nltk.word_tokenize(text)
    ps = PorterStemmer()
    return ' '.join([ps.stem(word) for word in tokens if word not in stopwords.words('english')])

# TF-IDF Vectorization
def tfidf_vectorize(corpus):
    vectorizer = TfidfVectorizer(max_features=500)
    return vectorizer.fit_transform(corpus)

# BM25 Ranking
def bm25_ranking(corpus):
    tokenized_corpus = [doc.split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)

