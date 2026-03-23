from .word_actions import *
from .constants import CACHE_DIR, BM25_K1, BM25_B, RETURN_DOCUMENT_LIMIT

import pickle, os, collections, math


class InvertedIndex:
    def __init__(self):
        self.index = collections.defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, 'index.pkl')
        self.docmap_path = os.path.join(CACHE_DIR, 'docmap.pkl')
        self.term_frequencies = collections.defaultdict(collections.Counter)
        self.freq_path = os.path.join(CACHE_DIR, 'term_frequencies.pkl')
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, 'doc_lengths.pkl')

    def __add_document(self, doc_id, text):
        text = separator(text)
        word_list = []
        for word in text:
            word_list.append(word)
            if word not in self.index:
                self.index[word] = {doc_id}
            else:
                self.index[word].add(doc_id)
        self.term_frequencies[doc_id].update(word_list)
        self.doc_lengths[doc_id] = len(word_list)

    def __get_avg_doc_length(self) -> float: 
        #CH9.4: replacing with solution code; is this where it's eating shit??
        '''if len(self.doc_lengths) > 0:
            return sum(self.doc_lengths.values()) / len(self.doc_lengths)
        return 0.0'''
        if not self.doc_lengths or len(self.doc_lengths) == 0:
            return 0.0
        total_length = 0
        for length in self.doc_lengths.values():
            total_length += length
        return total_length / len(self.doc_lengths)
        

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

    """ def get_df(self, term): #i don't know what the purpose of this was but it shouldn't be necessary
        tokens = separator(term)
        if len(tokens) > 1:
            raise Exception('multiple terms unsupported')
        if tokens[0] in self.index.keys():
            return len(self.index[tokens[0]]) 
        return 0 """

    def get_tf(self, doc_id, term):
        term = separator(term)
        if len(term) > 1:
            raise Exception('multiple terms unsupported')
        term_counts = self.term_frequencies[doc_id]
        return term_counts[term[0]]
        
    def get_idf(self, term):
        tokens = separator(term)
        if len(tokens) != 1:
            raise ValueError('one term per idf')
        token = tokens[0]
        num_docs = len(self.docmap)
        if token in self.index.keys():
            doc_freq = len(self.index[token])
        else:
            doc_freq = 0
        return math.log((num_docs + 1) / (doc_freq + 1))
    
    def get_bm25_idf(self, term: str) -> float:
        tokens = separator(term)
        if len(tokens) > 1:
            raise Exception('multiple terms unsupported')
        token = tokens[0]
        num_docs = len(self.docmap)
        if token in self.index.keys():
            doc_freq = len(self.index[token])
        else:
            doc_freq = 0
        return math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1) #math matches the solution files as of chapter 9
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B): 
        #CH9.4 updated; i think this just safeguards against /0 errors?
        tf = self.get_tf(doc_id, term)
        this_doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (this_doc_length / avg_doc_length)
        else:
            length_norm = 1
        #print(f"DEBUG: document length is {self.doc_lengths[doc_id]}, average is {avg_doc_length}, length norm should be {length_norm}.")
        term_freq = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        #print(f"DEBUG: term frequency should be {term_freq}. calculated as ({tf} * ({k1} + 1 )) / ({tf} + {k1} * {length_norm}).")
        return term_freq
    
    def bm25(self, doc_id, term):
        #CH9.4 looks the exact same
        idf = self.get_bm25_idf(term)
        tf = self.get_bm25_tf(doc_id, term)
        return tf * idf
    
    def bm25_search(self, query, limit=5, debug = False):
        tokens = separator(query)
        if debug:
            print(f"DEBUG: tokens: {tokens}")
        score_matches = {}
        for document in self.docmap:
            score = 0.0
            #CH9.4 cleaned up to match solution. using a += saves you running into key errors
            for token in tokens:
                score += self.bm25(document, token)
            score_matches[document] = score
            '''if document not in score_matches:
                score_matches[document] = 0
            for token in tokens:
                #print(token)
                score_matches[document] += self.bm25(document, token)
                #print(score_matches[document])'''
        #print(score_matches)
        sorted_scores = sorted(score_matches.items(), key=lambda item: item[1], reverse=True)
        
        #okay fuck this apparently i fucked it
        #return {k: v for i, (k, v) in enumerate(score_matches.items()) if i < limit}
        results = []
        for doc_id, score in sorted_scores[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(doc['id'], doc['title'], doc['description'][:RETURN_DOCUMENT_LIMIT], score)
            if debug: 
                print(f"DEBUG: bm25 result: {formatted_result}")
                kg = input("Press Enter to continue...")
                if kg:
                    debug = False
            results.append(formatted_result)
        
        return results
