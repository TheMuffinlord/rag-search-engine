import string, json

from nltk.stem import PorterStemmer
from cli.lib.constants import DATA_PATH, STOPWORDS_PATH

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