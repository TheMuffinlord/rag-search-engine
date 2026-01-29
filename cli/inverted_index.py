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
        else:
            return set()
        
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

    def load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.docmap_path):
            with open(self.index_path, 'rb') as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, 'rb') as f:
                self.docmap = pickle.load(f)
        else:
            print(f"debug: path missing. index: {os.path.exists(self.index_path)}; docmap: {os.path.exists(self.docmap_path)}")
            raise Exception('uninitialized movie index')

            