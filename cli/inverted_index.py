from word_actions import *
from constants import CACHE_DIR

import pickle, os


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.index_path = os.path.join(CACHE_DIR, 'index.pkl')
        self.docmap_path = os.path.join(CACHE_DIR, 'docmap.pkl')

    def __add_document(self, doc_id, text):
        text = separator(text)
        for word in text:
            if word not in self.index:
                self.index[word] = {doc_id}
            else:
                self.index[word].add(doc_id)

    def get_documents(self, term: str):
        if term.lower() in self.index:
            return sorted(self.index[term.lower()])
        
    def build(self):
        movieList = load_movies()
        for movie in movieList:
            self.__add_document(movie['id'], f"{movie['title']} {movie['description']}")
            self.docmap[movie['id']] = movie

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)
            