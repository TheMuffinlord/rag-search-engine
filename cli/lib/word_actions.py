import string, json

from nltk.stem import PorterStemmer
from typing import Any
from .constants import DATA_PATH, STOPWORDS_PATH, SCORE_PRECISION

def stripper(term: str):
    return term.translate(str.maketrans('','', string.punctuation)).lower()

def separator(term: str) -> list: 
    term = stripper(term)
    terms = term.split()
    stopwords = load_stopwords()
    valid_terms = []
    for t in terms:
        if t and t not in stopwords:
            valid_terms.append(stemmer(t))
    return valid_terms

def stemmer(term: str):
    stemmer = PorterStemmer()
    return stemmer.stem(term)

def load_stopwords():
    with open(STOPWORDS_PATH) as f:
        wordList = f.read()
    return wordList.splitlines()

def load_movies():
    with open(DATA_PATH, 'r') as f:
        movieList = json.load(f) 
    return movieList["movies"]

def match_tokens(query_tokens, title_tokens):
    for query in query_tokens:
        for title in title_tokens:
            if query in title:
                return True
    return False

#i am going to copy this because i am going to lose my fuckign mind
def format_search_result(doc_id: str, title: str, document: str, score: float, **metadata: Any):
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }