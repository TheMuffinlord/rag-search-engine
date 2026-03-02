from .word_actions import *
from .constants import CACHE_DIR, BM25_K1, BM25_B
from itertools import islice

import pickle, os, collections, math


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.index_path = os.path.join(CACHE_DIR, 'index.pkl')
        self.docmap_path = os.path.join(CACHE_DIR, 'docmap.pkl')
        self.term_frequencies = {}
        self.freq_path = os.path.join(CACHE_DIR, 'term_frequencies.pkl')
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, 'doc_lengths.pkl')

    def __add_document(self, doc_id, text):
        text = separator(text)
        word_list = []
        self.doc_lengths[doc_id] = len(text)
        for word in text:
            word_list.append(word)
            if word not in self.index:
                self.index[word] = {doc_id}
            else:
                self.index[word].add(doc_id)
        self.term_frequencies[doc_id] = collections.Counter(word_list)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) > 0:
            return sum(self.doc_lengths.values()) / len(self.doc_lengths)
        return 0.0
        

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
        with (open(self.doc_lengths_path, 'wb')) as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.docmap_path) and os.path.exists(self.freq_path) and os.path.exists(self.doc_lengths_path):
            # how many of these are we going to have to do, do you reckon? should peek the solution notes to see if there's a better way
            with open(self.index_path, 'rb') as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, 'rb') as f:
                self.docmap = pickle.load(f)
            with open(self.freq_path, 'rb') as f:
                self.term_frequencies = pickle.load(f)         
            with open(self.doc_lengths_path, 'rb') as f:
                self.doc_lengths = pickle.load(f)
        else:
            raise Exception('uninitialized movieDB index')

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
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_doc_length)
        #print(f"DEBUG: document length is {self.doc_lengths[doc_id]}, average is {avg_doc_length}, length norm should be {length_norm}.")
        term_freq = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        #print(f"DEBUG: term frequency should be {term_freq}. calculated as ({tf} * ({k1} + 1 )) / ({tf} + {k1} * {length_norm}).")
        return term_freq
    
    def bm25(self, doc_id, term):
        idf = self.get_bm25_idf(term)
        tf = self.get_bm25_tf(doc_id, term)
        return tf * idf
    
    def bm25_search(self, query, limit=5) -> dict:
        tokens = separator(query)
        score_matches = {}
        for document in self.docmap:
            #print(document)
            if document not in score_matches:
                score_matches[document] = 0
            for token in tokens:
                #print(token)
                score_matches[document] += self.bm25(document, token)
                #print(score_matches[document])
        #print(score_matches)
        score_matches = {k: v for k, v in sorted(score_matches.items(), key=lambda item: item[1], reverse=True)}
        #print(score_matches)
        return {k: v for i, (k, v) in enumerate(score_matches.items()) if i < limit}