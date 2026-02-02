from word_actions import *
from constants import CACHE_DIR

import pickle, os, collections, math


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.index_path = os.path.join(CACHE_DIR, 'index.pkl')
        self.docmap_path = os.path.join(CACHE_DIR, 'docmap.pkl')
        self.term_frequencies = {}
        self.freq_path = os.path.join(CACHE_DIR, 'term_frequencies.pkl')

    def __add_document(self, doc_id, text):
        text = separator(text)
        word_list = []
        for word in text:
            word_list.append(word)
            if word not in self.index:
                self.index[word] = {doc_id}
            else:
                self.index[word].add(doc_id)
        self.term_frequencies[doc_id] = collections.Counter(word_list)

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
        with (open(self.freq_path, 'wb')) as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.docmap_path) and os.path.exists(self.freq_path):
            with open(self.index_path, 'rb') as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, 'rb') as f:
                self.docmap = pickle.load(f)
            with open(self.freq_path, 'rb') as f:
                self.term_frequencies = pickle.load(f)         
        else:
            print(f"debug: path missing. index: {os.path.exists(self.index_path)}; docmap: {os.path.exists(self.docmap_path)}")
            raise Exception('uninitialized movie index')

    def get_df(self, term):
        term = separator(term)
        if len(term) > 1:
            raise Exception('multiple terms unsupported')
        return len(self.index[term[0]])

    def get_tf(self, doc_id, term):
        term = separator(term)
        if len(term) > 1:
            raise Exception('multiple terms unsupported')
        term_counts = self.term_frequencies[doc_id]
        return term_counts[term[0]]
        
    def get_bm25_idf(self, term: str) -> float:
        term = separator(term)
        if len(term) > 1:
            raise Exception('multiple terms unsupported')
        num_docs = len(self.docmap)
        doc_freq = self.get_df(term[0])
        return math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
